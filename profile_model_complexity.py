"""Profile model complexity and optional segmentation Dice.

This script is intended for the complexity table in the paper:

    Params | FLOPs (G) | Inference time (ms) | Dice

Notes:
    * Params and FLOPs depend only on the architecture and input shape.
    * Inference time should be measured with the same checkpoint/config used
      for final testing, because kernels and branches should match the real run.
    * Dice requires a checkpoint and a validation/test split.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.losses import unpack_model_output  # noqa: E402
from train.train_config import (  # noqa: E402
    build_model_from_config,
    checkpoint_name_from_config,
    load_config,
    resolve_config_path,
)
from train.train_kpta import build_dataset  # noqa: E402
from visualize_kpta_25d_val import extract_state_dict  # noqa: E402


def parse_shape(shape_text: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse a comma-separated input shape string such as '1,3,17,256,256'."""
    if not shape_text:
        return None
    shape = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
    if not shape:
        raise ValueError("--input_shape was provided but no dimensions were parsed")
    return shape


def infer_dummy_shape(config: Dict, batch_size: int) -> Tuple[int, ...]:
    """Infer a dummy input shape from the YAML config.

    KPTA-Net uses [B,C,H,W], while KPTA-2.5DNet uses [B,K,T,H,W].
    """
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    model_name = model_cfg.get("name")
    image_size = int(dataset_cfg.get("img_size", 256))

    if model_name == "kpta_25d_net":
        num_slices = int(model_cfg.get("num_slices", 3))
        in_phases = int(model_cfg.get("in_phases", 17))
        return (batch_size, num_slices, in_phases, image_size, image_size)

    if model_name == "kpta_net":
        in_channels = int(model_cfg.get("in_channels", 17))
        return (batch_size, in_channels, image_size, image_size)

    raise ValueError(
        "Cannot infer dummy shape for model.name="
        f"{model_name!r}. Use --input_shape explicitly."
    )


def resolve_split_path(config: Dict, split: str, override: Optional[str]) -> Optional[Path]:
    """Resolve train/val/test split path from CLI override or config."""
    if override:
        return Path(override)
    train_cfg = config.get("train", {})
    key = f"{split}_path"
    path = train_cfg.get(key)
    return Path(path) if path else None


def build_loader(config: Dict, split_path: Path, batch_size: int, num_workers: int):
    """Build the project dataset loader used for optional Dice and timing."""
    dataset_cfg = config.get("dataset", {})
    dataset_type = dataset_cfg.get("type", "breastdm_2d")
    input_phase_indices = dataset_cfg.get("input_phase_indices")
    dataset = build_dataset(
        str(split_path),
        dataset_type=dataset_type,
        img_size=int(dataset_cfg.get("img_size", 256)),
        input_phase_indices=input_phase_indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def extract_logits(output):
    """Support tensor outputs and dict outputs from research models."""
    if isinstance(output, dict):
        if "seg_logits" in output:
            return output["seg_logits"]
        if "logits" in output:
            return output["logits"]
    return unpack_model_output(output)


def profile_flops(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    use_amp: bool,
) -> int:
    """Estimate FLOPs with PyTorch profiler.

    PyTorch reports FLOPs for supported operations such as Conv2d and matmul.
    Attention softmax, normalization and some custom ops may be under-counted,
    so the value should be reported as an implementation-level estimate.
    """
    device = example_input.device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                output = model(example_input, return_dict=True)
                _ = extract_logits(output)
            if device.type == "cuda":
                torch.cuda.synchronize()

    return int(sum(event.flops or 0 for event in profiler.key_averages()))


def measure_inference_time(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    warmup: int,
    repeat: int,
    use_amp: bool,
) -> Tuple[float, float]:
    """Measure average and standard deviation inference time per sample in ms."""
    device = example_input.device
    batch_size = int(example_input.shape[0])

    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                _ = extract_logits(model(example_input, return_dict=True))
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = []
        for _ in range(max(1, repeat)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                _ = extract_logits(model(example_input, return_dict=True))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000.0 / batch_size)

    mean_ms = float(statistics.mean(elapsed))
    std_ms = float(statistics.pstdev(elapsed)) if len(elapsed) > 1 else 0.0
    return mean_ms, std_ms


def dice_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> float:
    """Compute batch mean Dice from segmentation logits and binary masks."""
    probs = torch.sigmoid(logits.float())
    preds = (probs >= threshold).float()
    masks = (masks.float() >= 0.5).float().to(preds.device)
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * masks).sum(dim=dims)
    denominator = preds.sum(dim=dims) + masks.sum(dim=dims)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return float(dice.mean().item())


def evaluate_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    max_batches: Optional[int],
) -> float:
    """Evaluate mean Dice on a split. Use max_batches for quick profiling runs."""
    scores = []
    with torch.inference_mode():
        for batch_index, (images, masks, _names) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = extract_logits(model(images, return_dict=True))
            scores.append(dice_from_logits(logits, masks, threshold))
    return float(np.mean(scores)) if scores else float("nan")


def load_checkpoint_if_available(
    model: torch.nn.Module,
    checkpoint_path: Optional[Path],
    device: torch.device,
) -> bool:
    """Load checkpoint when provided. Return True if weights were loaded."""
    if checkpoint_path is None:
        return False
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    return True


def first_batch_or_dummy(
    loader: Optional[DataLoader],
    dummy_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Use a real batch for timing when available; otherwise create zeros."""
    if loader is not None:
        images, _masks, _names = next(iter(loader))
        return images.to(device, non_blocking=True)
    return torch.zeros(dummy_shape, dtype=torch.float32, device=device)


def write_csv(path: Path, row: Dict[str, object]) -> None:
    """Append one profiling row to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile Params, FLOPs, inference time and optional Dice."
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path. Default uses <train.output_path>/"
            "best_model_<config-name>.pth when --auto_checkpoint is set."
        ),
    )
    parser.add_argument(
        "--auto_checkpoint",
        action="store_true",
        help="Resolve checkpoint from train.output_path and config file name.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--input_shape",
        default=None,
        help="Optional dummy shape, e.g. 1,3,17,256,256. Overrides config shape.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_dice_batches",
        type=int,
        default=None,
        help="Limit Dice evaluation batches for a quick run. Default: full split.",
    )
    parser.add_argument(
        "--no_dice",
        action="store_true",
        help="Skip Dice evaluation even if a checkpoint and split are available.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA autocast during timing/evaluation.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path to append the profiling result.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is None and args.auto_checkpoint:
        output_path = Path(config.get("train", {}).get("output_path", "."))
        checkpoint_path = output_path / checkpoint_name_from_config(config_path)

    model = build_model_from_config(config).to(device)
    loaded_checkpoint = load_checkpoint_if_available(model, checkpoint_path, device)
    model.eval()

    split_path = resolve_split_path(config, args.split, args.split_path)
    loader = None
    dataset_size = 0
    if split_path is not None and split_path.exists():
        dataset, loader = build_loader(
            config,
            split_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        dataset_size = len(dataset)

    dummy_shape = parse_shape(args.input_shape) or infer_dummy_shape(
        config,
        args.batch_size,
    )
    example_input = first_batch_or_dummy(loader, dummy_shape, device)

    total_params, trainable_params = count_parameters(model)
    flops = profile_flops(model, example_input, use_amp=use_amp)
    time_ms, time_std_ms = measure_inference_time(
        model,
        example_input,
        warmup=args.warmup,
        repeat=args.repeat,
        use_amp=use_amp,
    )

    dice = float("nan")
    if not args.no_dice and loaded_checkpoint and loader is not None:
        dice = evaluate_dice(
            model,
            loader,
            device=device,
            threshold=args.threshold,
            use_amp=use_amp,
            max_batches=args.max_dice_batches,
        )

    row = {
        "config": Path(config_path).name,
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "input_shape": "x".join(str(dim) for dim in tuple(example_input.shape)),
        "params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "flops_g": flops / 1e9,
        "inference_time_ms": time_ms,
        "inference_time_std_ms": time_std_ms,
        "dice": dice,
        "threshold": args.threshold,
        "split": args.split,
        "dataset_size": dataset_size,
        "device": str(device),
        "amp": use_amp,
    }

    print("Complexity profile")
    print(f"  config: {row['config']}")
    print(f"  checkpoint loaded: {loaded_checkpoint}")
    print(f"  input shape: {row['input_shape']}")
    print(f"  Params: {row['params_m']:.3f} M")
    print(f"  Trainable Params: {row['trainable_params_m']:.3f} M")
    print(f"  FLOPs: {row['flops_g']:.3f} G")
    print(
        "  Inference time: "
        f"{row['inference_time_ms']:.3f} +/- "
        f"{row['inference_time_std_ms']:.3f} ms/sample"
    )
    if np.isfinite(dice):
        print(f"  Dice: {dice:.4f}")
    else:
        print("  Dice: skipped")
    print(
        "  Note: FLOPs are PyTorch profiler estimates for supported ops; "
        "attention/normalization may be partially under-counted."
    )

    if args.csv:
        write_csv(Path(args.csv), row)
        print(f"Saved CSV row to: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
