import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce_msdahnet.datasets.breastdm_2d_dataset import BreastDM2DDataset, collect_split_pairs  # noqa: E402
from reproduce_msdahnet.models.msdahnet import build_msdahnet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BreastDM fixed split tensor shapes for MSDAHNet.")
    parser.add_argument("--config", default="reproduce_msdahnet/configs/msdahnet_breastdm_fixed_split.yaml")
    parser.add_argument("--save-preview", action="store_true", help="Save selected channel/mask previews.")
    parser.add_argument(
        "--strict-patient-split",
        action="store_true",
        help="Raise an error when the fixed split contains overlapping patient IDs.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    expected_channels = int(cfg["model"]["in_channels"])
    print("=== MSDAHNet BreastDM Fixed Split Inspection ===")
    print(f"config: {resolve_path(args.config)}")
    print(f"input_mode: {cfg['data'].get('input_mode', 'single_channel_pre')}")
    print(f"data.in_channels: {cfg['data'].get('in_channels')}")
    print(f"model.in_channels: {expected_channels}")
    print(f"image_size: {cfg['data']['image_size']}")
    print(f"label_phase: {cfg['data'].get('label_phase', 'unknown')}")

    split_pairs: Dict[str, List[Tuple[str, str]]] = {
        "train": collect_split_pairs(resolve_path(cfg["data"]["train_path"])),
        "val": collect_split_pairs(resolve_path(cfg["data"]["val_path"])),
        "test": collect_split_pairs(resolve_path(cfg["data"]["test_path"])),
    }
    report_patient_leakage(split_pairs, strict=args.strict_patient_split)

    preview_root = Path(resolve_path(cfg["output"]["output_dir"])) / "debug_previews"
    for split, pairs in split_pairs.items():
        inspect_split(split, pairs, cfg, expected_channels, preview_root if args.save_preview else None)

    model = build_msdahnet(in_channels=expected_channels, num_classes=int(cfg["model"]["num_classes"]))
    first_conv = next((m for m in model.modules() if isinstance(m, torch.nn.Conv2d)), None)
    if first_conv is None:
        raise RuntimeError("No Conv2d layer found in MSDAHNet.")
    print(f"model_first_conv_in_channels: {first_conv.in_channels}")
    if int(first_conv.in_channels) != expected_channels:
        raise ValueError(f"Model first conv expects {first_conv.in_channels}, config expects {expected_channels}.")
    print("inspection_status: OK")


def inspect_split(split: str, pairs: List[Tuple[str, str]], cfg, expected_channels: int, preview_root: Path = None) -> None:
    print(f"\n--- {split} ---")
    print(f"pairs: {len(pairs)}")
    first_image, first_mask = pairs[0]
    print(f"first_image: {first_image}")
    print(f"first_mask: {first_mask}")
    print(f"raw_image_shape: {raw_shape(first_image)}")
    print(f"raw_mask_shape: {raw_shape(first_mask)}")

    dataset = BreastDM2DDataset(
        pairs,
        image_size=int(cfg["data"]["image_size"]),
        gray_to_rgb=bool(cfg["data"].get("gray_to_rgb", False)),
        mask_threshold=float(cfg["data"]["mask_threshold"]),
        input_mode=cfg["data"].get("input_mode", "single_channel_pre"),
        channel_index=cfg["data"].get("channel_index", None),
    )
    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    print(f"dataset_image_shape: {tuple(image.shape)}")
    print(f"dataset_mask_shape: {tuple(mask.shape)}")
    print(f"image_min_mean_max: {float(image.min()):.6f} / {float(image.mean()):.6f} / {float(image.max()):.6f}")
    print(f"mask_positive_ratio_first: {float(mask.mean()):.6f}")
    if int(image.shape[0]) != expected_channels:
        raise ValueError(f"{split}: Dataset C={image.shape[0]}, model expects C={expected_channels}.")

    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"batch_image_shape: {tuple(batch['image'].shape)}")
    print(f"batch_mask_shape: {tuple(batch['mask'].shape)}")

    pos_ratios = []
    channel_means = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        pos_ratios.append(float(item["mask"].mean()))
        if idx < 32:
            channel_means.append(item["image"].mean(dim=(1, 2)).numpy())
    print(f"mask_positive_samples: {sum(v > 0 for v in pos_ratios)} / {len(pos_ratios)}")
    print(f"mask_positive_ratio_mean_min_max: {np.mean(pos_ratios):.6f} / {np.min(pos_ratios):.6f} / {np.max(pos_ratios):.6f}")
    if channel_means:
        mean_vec = np.stack(channel_means, axis=0).mean(axis=0)
        preview_len = min(len(mean_vec), 17)
        print(f"channel_mean_first_{preview_len}: {np.array2string(mean_vec[:preview_len], precision=4, separator=', ')}")

    if preview_root is not None:
        save_preview(preview_root / split, image.numpy(), mask.numpy()[0])


def report_patient_leakage(split_pairs: Dict[str, List[Tuple[str, str]]], strict: bool = False) -> None:
    patient_sets = {split: {patient_id(path) for path, _ in pairs} for split, pairs in split_pairs.items()}
    print("\n--- patient split ---")
    for split, patients in patient_sets.items():
        print(f"{split}_patients: {len(patients)}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[a] & patient_sets[b]
        print(f"{a}_{b}_patient_overlap: {len(overlap)}")
        if overlap:
            examples = sorted(overlap)[:5]
            message = f"Patient overlap detected between {a} and {b}: {examples}"
            if strict:
                raise ValueError(message)
            print(f"WARNING: {message}")


def patient_id(path: str) -> str:
    stem = Path(path).stem
    if "_p-" in stem:
        return stem.split("_p-", 1)[0]
    return stem.split("_", 1)[0]


def raw_shape(path: str):
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        return tuple(np.load(str(path_obj), mmap_mode="r").shape)
    image = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
    return None if image is None else tuple(image.shape)


def save_preview(out_dir: Path, image: np.ndarray, mask: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for channel in preview_channels(image.shape[0]):
        channel_image = image[channel]
        png = (channel_image * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"channel_{channel:02d}.png"), png)
    cv2.imwrite(str(out_dir / "mask.png"), (mask * 255.0).clip(0, 255).astype(np.uint8))


def preview_channels(num_channels: int) -> Iterable[int]:
    candidates = [0, 1, 8, 9, 16]
    return [idx for idx in candidates if idx < num_channels]


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def load_config(path: str):
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
