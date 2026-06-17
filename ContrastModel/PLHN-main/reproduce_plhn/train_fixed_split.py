from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PLHN_ROOT = THIS_DIR.parent
if str(PLHN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLHN_ROOT))

from Model.TokenSegV8_prototype_fusion_attentions import TokenSegV8  # noqa: E402
from reproduce_plhn.prepare_breastdm_3d import convert_dataset  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    candidate = PLHN_ROOT / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {path}")


def resolve_path(path: str, base: Path = PLHN_ROOT) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return base / p


def read_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def minmax_normalize(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - vmin) / (vmax - vmin)


def load_case(case_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    pre = minmax_normalize(read_volume(case_dir / "P0.nii.gz"))
    post = minmax_normalize(read_volume(case_dir / "P1.nii.gz"))
    target = (read_volume(case_dir / "GT.nii.gz") > 0).astype(np.float32)
    if pre.shape != post.shape or pre.shape != target.shape:
        raise ValueError(
            f"Case {case_dir} shape mismatch: P0={pre.shape}, P1={post.shape}, GT={target.shape}"
        )
    subtraction = post - pre
    image = np.stack([post, subtraction], axis=0).astype(np.float32)
    return image, target[None].astype(np.float32)


def read_entries(processed_root: Path, split: str, fold: int) -> List[Path]:
    list_path = processed_root / "data_folder" / f"{split}{fold}.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing split file: {list_path}")
    entries = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [processed_root / entry for entry in entries]


def pad_to_patch(array: np.ndarray, patch_size: Sequence[int]) -> np.ndarray:
    spatial = array.shape[-3:]
    pad_width = [(0, 0)] * (array.ndim - 3)
    for size, patch in zip(spatial, patch_size):
        missing = max(int(patch) - int(size), 0)
        before = missing // 2
        after = missing - before
        pad_width.append((before, after))
    if any(before or after for before, after in pad_width):
        return np.pad(array, pad_width, mode="constant")
    return array


def crop_patch(
    image: np.ndarray,
    target: np.ndarray,
    patch_size: Sequence[int],
    rng: np.random.Generator,
    positive_crop_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    image = pad_to_patch(image, patch_size)
    target = pad_to_patch(target, patch_size)
    dims = image.shape[-3:]

    use_positive = rng.random() < positive_crop_prob and target.sum() > 0
    if use_positive:
        coords = np.argwhere(target[0] > 0)
        center = coords[int(rng.integers(0, len(coords)))]
        starts = []
        for axis, patch in enumerate(patch_size):
            low = int(center[axis]) - int(patch) + 1
            high = int(center[axis])
            start = int(rng.integers(low, high + 1)) if high >= low else int(center[axis]) - int(patch) // 2
            start = max(0, min(start, int(dims[axis]) - int(patch)))
            starts.append(start)
    else:
        starts = [
            int(rng.integers(0, int(size) - int(patch) + 1)) if int(size) > int(patch) else 0
            for size, patch in zip(dims, patch_size)
        ]

    z, y, x = starts
    dz, dy, dx = [int(v) for v in patch_size]
    return image[:, z : z + dz, y : y + dy, x : x + dx], target[:, z : z + dz, y : y + dy, x : x + dx]


class PLHNVolumePatchDataset(Dataset):
    def __init__(
        self,
        case_dirs: Sequence[Path],
        patch_size: Sequence[int],
        samples_per_volume: int,
        positive_crop_prob: float,
        seed: int,
    ) -> None:
        self.case_dirs = list(case_dirs)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_volume = int(samples_per_volume)
        self.positive_crop_prob = float(positive_crop_prob)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.case_dirs) * self.samples_per_volume

    def __getitem__(self, index: int):
        case_dir = self.case_dirs[index % len(self.case_dirs)]
        image, target = load_case(case_dir)
        image, target = crop_patch(image, target, self.patch_size, self.rng, self.positive_crop_prob)
        return torch.from_numpy(image), torch.from_numpy(target), case_dir.name


def dice_loss(probability: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = probability.contiguous()
    target = target.contiguous()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def resize_target(target: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
    if tuple(target.shape[-3:]) == tuple(spatial_size):
        return target
    return F.interpolate(target.float(), size=tuple(spatial_size), mode="nearest")


def unpack_outputs(outputs):
    if isinstance(outputs, (tuple, list)):
        if len(outputs) != 4:
            raise ValueError(f"Expected 4 PLHN outputs, got {len(outputs)}")
        return outputs
    raise ValueError("PLHN training output must be a tuple/list of 4 tensors.")


def compute_loss(outputs, target: torch.Tensor, loss_cfg: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
    proto_output, proto_prob, raw_prob, fusion_prob = unpack_outputs(outputs)
    components: Dict[str, torch.Tensor] = {}

    for name, probability in (
        ("fusion", fusion_prob),
        ("raw", raw_prob),
        ("prototype", proto_prob),
    ):
        branch_target = resize_target(target, probability.shape[-3:])
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        components[f"{name}_bce"] = F.binary_cross_entropy(probability, branch_target)
        components[f"{name}_dice"] = dice_loss(probability, branch_target)

    if isinstance(proto_output, dict) and float(loss_cfg.get("contrast_weight", 0.0)) > 0:
        seg = proto_output["seg"]
        ce_target = resize_target(target, seg.shape[-3:]).squeeze(1).long()
        components["proto_ce"] = F.cross_entropy(seg, ce_target, ignore_index=-1)

        logits = proto_output.get("logits")
        contrast_target = proto_output.get("target")
        if logits is not None and contrast_target is not None:
            contrast_target = contrast_target.long()
            valid = contrast_target != -1
            if bool(valid.any()):
                components["proto_ppc"] = F.cross_entropy(logits, contrast_target, ignore_index=-1)
                gathered = torch.gather(logits[valid], 1, contrast_target[valid, None])
                components["proto_ppd"] = (1.0 - gathered).pow(2).mean()

    weights = {
        "fusion_bce": float(loss_cfg.get("fusion_bce_weight", 1.0)),
        "fusion_dice": float(loss_cfg.get("fusion_dice_weight", 1.0)),
        "raw_bce": float(loss_cfg.get("raw_bce_weight", 1.0)),
        "raw_dice": float(loss_cfg.get("raw_dice_weight", 8.0)),
        "prototype_bce": float(loss_cfg.get("prototype_bce_weight", 1.0)),
        "prototype_dice": float(loss_cfg.get("prototype_dice_weight", 1.5)),
        "proto_ce": float(loss_cfg.get("contrast_weight", 0.3)),
        "proto_ppc": float(loss_cfg.get("ppc_weight", 0.003)),
        "proto_ppd": float(loss_cfg.get("ppd_weight", 0.0003)),
    }

    total = sum(components[name] * weights.get(name, 1.0) for name in components)
    log_values = {name: float(value.detach().cpu()) for name, value in components.items()}
    log_values["loss"] = float(total.detach().cpu())
    return total, log_values


def starts_for_dim(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, max(1, stride)))
    final = size - patch
    if starts[-1] != final:
        starts.append(final)
    return starts


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return autocast(enabled=True)
    return nullcontext()


def extract_probability(outputs: object, patch_size: Sequence[int]) -> torch.Tensor:
    if isinstance(outputs, (tuple, list)):
        probability = outputs[-1]
    else:
        probability = outputs
    if tuple(probability.shape[-3:]) != tuple(patch_size):
        probability = F.interpolate(probability, size=tuple(patch_size), mode="trilinear", align_corners=False)
    return probability


def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: Sequence[int],
    stride: Sequence[int],
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    original_shape = image.shape[-3:]
    padded = pad_to_patch(image, patch_size)
    padded_shape = padded.shape[-3:]
    output = torch.zeros((1, 1, *padded_shape), dtype=torch.float32, device=device)
    counts = torch.zeros_like(output)
    starts = [starts_for_dim(int(size), int(patch), int(step)) for size, patch, step in zip(padded_shape, patch_size, stride)]

    model.eval()
    with torch.inference_mode():
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    patch = padded[:, z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]]
                    patch_tensor = torch.from_numpy(patch[None]).float().to(device, non_blocking=True)
                    with autocast_context(device, amp):
                        probability = extract_probability(model(patch_tensor), patch_size)
                    output[:, :, z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]] += probability
                    counts[:, :, z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]] += 1.0
    output = (output / counts.clamp_min(1.0)).cpu().numpy()[0, 0]
    d, h, w = original_shape
    return output[:d, :h, :w]


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = (prediction > 0).astype(np.uint8)
    gt = (target > 0).astype(np.uint8)
    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())
    tn = float(((1 - pred) * (1 - gt)).sum())
    eps = 1e-8
    if pred.sum() == 0 and gt.sum() == 0:
        return {"dice": 1.0, "iou": 1.0, "recall": 1.0, "precision": 1.0, "accuracy": 1.0}
    return {
        "dice": (2.0 * tp) / (2.0 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "recall": tp / (tp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def mean_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"dice": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0, "accuracy": 0.0}
    keys = ("dice", "iou", "recall", "precision", "accuracy")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate_split(
    model: torch.nn.Module,
    case_dirs: Sequence[Path],
    split: str,
    cfg: Dict,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    eval_cfg = cfg.get("eval", {})
    patch_size = tuple(int(v) for v in cfg["train"]["patch_size"])
    stride = tuple(int(v) for v in eval_cfg.get("stride", patch_size))
    threshold = float(eval_cfg.get("threshold", 0.5))
    amp = bool(cfg["train"].get("amp", True))
    save_predictions = bool(cfg.get("output", {}).get("save_predictions", True))

    rows: List[Dict[str, float]] = []
    global_preds: List[np.ndarray] = []
    global_targets: List[np.ndarray] = []
    pred_dir = output_dir / "predicted_masks" / split
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for case_dir in tqdm(case_dirs, desc=f"{split}", leave=False):
        image, target = load_case(case_dir)
        probability = predict_volume(model, image, patch_size, stride, device, amp)
        prediction = probability >= threshold
        metrics = binary_metrics(prediction, target[0])
        rows.append({"case": case_dir.name, **metrics})
        global_preds.append(prediction.reshape(-1).astype(np.uint8))
        global_targets.append(target[0].reshape(-1).astype(np.uint8))
        if save_predictions:
            sitk.WriteImage(sitk.GetImageFromArray(prediction.astype(np.uint8)), str(pred_dir / f"{case_dir.name}.nii.gz"))

    global_metrics = (
        binary_metrics(np.concatenate(global_preds), np.concatenate(global_targets))
        if global_preds
        else mean_metrics([])
    )
    return {"mean": mean_metrics(rows), "global": global_metrics, "cases": rows}


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_prepare_data(cfg: Dict) -> Path:
    data_cfg = cfg["data"]
    processed_root = resolve_path(data_cfg["processed_root"])
    fold = int(data_cfg.get("fold", 1))
    required = [processed_root / "data_folder" / f"{split}{fold}.txt" for split in ("train", "val", "test")]
    if bool(data_cfg.get("convert_from_seg", False)) and (
        bool(data_cfg.get("force_convert", False)) or not all(path.exists() for path in required)
    ):
        summary = convert_dataset(
            seg_root=resolve_path(data_cfg["seg_root"]),
            output_root=processed_root,
            fold=fold,
            pre_phase=data_cfg.get("pre_phase", "VIBRANT"),
            post_phase=data_cfg.get("post_phase", "VIBRANT+C8"),
            label_phase=data_cfg.get("label_phase", "VIBRANT"),
            image_size=int(data_cfg.get("image_size", 256)),
            mask_threshold=int(data_cfg.get("mask_threshold", 0)),
        )
        print(f"[prepare] converted BreastDM seg data: {summary['counts']}", flush=True)
    return processed_root


def build_model(cfg: Dict, patch_size: Sequence[int], device: torch.device) -> TokenSegV8:
    model_cfg = cfg.get("model", {})
    model = TokenSegV8(
        inch=2,
        outch=1,
        imgsize=[int(v) for v in patch_size],
        base_channeel=int(model_cfg.get("base_channels", 16)),
        hidden_size=int(model_cfg.get("hidden_size", 192)),
        window_size=int(model_cfg.get("window_size", 4)),
        TransformerLayerNum=int(model_cfg.get("transformer_layers", 4)),
    ).to(device)
    model.use_prototype = bool(model_cfg.get("use_prototype_learning", True))
    return model


def train_one_epoch(model, loader, optimizer, scaler, device, cfg) -> Dict[str, float]:
    model.train()
    amp = bool(cfg["train"].get("amp", True))
    loss_cfg = cfg.get("loss", {})
    totals: Dict[str, float] = {}
    count = 0
    for images, targets, _ in tqdm(loader, desc="train", leave=False):
        images = images.float().to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            outputs = model(images, targets)
            loss, parts = compute_loss(outputs, targets, loss_cfg)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(images.shape[0])
        count += batch_size
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: Dict, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": cfg,
        },
        str(path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PLHN on BreastDM fixed train/val/test split.")
    parser.add_argument("--config", default="reproduce_plhn/configs/plhn_breastdm_3d.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))

    processed_root = maybe_prepare_data(cfg)
    fold = int(cfg["data"].get("fold", 1))
    train_cases = read_entries(processed_root, "train", fold)
    val_cases = read_entries(processed_root, "val", fold)
    test_cases = read_entries(processed_root, "test", fold)

    device_name = cfg.get("experiment", {}).get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = resolve_path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    patch_size = tuple(int(v) for v in cfg["train"]["patch_size"])
    train_dataset = PLHNVolumePatchDataset(
        train_cases,
        patch_size=patch_size,
        samples_per_volume=int(cfg["train"].get("samples_per_volume", 4)),
        positive_crop_prob=float(cfg["train"].get("positive_crop_prob", 0.7)),
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"].get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    print(
        f"[PLHN] train={len(train_cases)} val={len(val_cases)} test={len(test_cases)} "
        f"patch={patch_size} device={device}",
        flush=True,
    )
    model = build_model(cfg, patch_size, device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and bool(cfg["train"].get("amp", True))))
    scheduler = None
    if cfg.get("scheduler", {}).get("enabled", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg["scheduler"].get("factor", 0.5)),
            patience=int(cfg["scheduler"].get("patience", 8)),
            min_lr=float(cfg["scheduler"].get("min_lr", 1e-6)),
        )

    best_dice = -1.0
    log_rows: List[Dict[str, object]] = []
    for epoch in range(1, int(cfg["train"].get("epochs", 100)) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        val_summary = evaluate_split(model, val_cases, "val", cfg, device, output_dir)
        val_dice = float(val_summary["mean"]["dice"])
        if scheduler is not None:
            scheduler.step(val_dice)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_summary["mean"].items()},
        }
        log_rows.append(row)
        write_csv(output_dir / "training_log.csv", log_rows)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, val_summary, cfg)
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, val_summary, cfg)
        print(
            f"[epoch {epoch:03d}] loss={train_metrics.get('loss', 0.0):.4f} "
            f"val_dice={val_dice:.4f} best={best_dice:.4f}",
            flush=True,
        )

    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        checkpoint = torch.load(str(best_path), map_location=device)
        model.load_state_dict(checkpoint["model"])
    test_summary = evaluate_split(model, test_cases, "test", cfg, device, output_dir)
    (output_dir / "metrics_test.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    print(
        "[test] "
        f"dice={test_summary['mean']['dice']:.4f} "
        f"iou={test_summary['mean']['iou']:.4f} "
        f"recall={test_summary['mean']['recall']:.4f} "
        f"precision={test_summary['mean']['precision']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
