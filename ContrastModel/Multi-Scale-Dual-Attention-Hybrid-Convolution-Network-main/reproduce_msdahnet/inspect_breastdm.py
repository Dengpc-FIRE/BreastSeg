import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce_msdahnet.datasets.breastdm_2d_dataset import BreastDM2DDataset, collect_pairs  # noqa: E402
from reproduce_msdahnet.metrics.segmentation_metrics import compute_sample_metrics  # noqa: E402
from reproduce_msdahnet.utils.split import infer_patient_id, make_folds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MSDAHNet BreastDM reproduction data.")
    parser.add_argument("--config", default="reproduce_msdahnet/configs/msdahnet_breastdm_5fold.yaml")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--max-preview", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_dir = resolve_path(cfg["data"]["image_dir"])
    mask_dir = resolve_path(cfg["data"]["mask_dir"])
    pairs = collect_pairs(image_dir, mask_dir)
    image_paths = [p[0] for p in pairs]
    patients = [infer_patient_id(p) for p in image_paths]
    folds, patient_level = make_folds(
        image_paths,
        n_splits=int(cfg["cross_validation"]["n_splits"]),
        split_level=cfg["cross_validation"]["split_level"],
        seed=int(cfg["cross_validation"]["seed"]),
        shuffle=bool(cfg["cross_validation"]["shuffle"]),
    )

    dataset = BreastDM2DDataset(
        pairs,
        image_size=int(cfg["data"]["image_size"]),
        gray_to_rgb=bool(cfg["data"].get("gray_to_rgb", False)),
        mask_threshold=float(cfg["data"]["mask_threshold"]),
    )
    stats = collect_dataset_stats(dataset)
    print(f"image_dir: {image_dir}")
    print(f"mask_dir: {mask_dir}")
    print(f"paired_samples: {len(pairs)}")
    print(f"unique_patients: {len(set(patients))}")
    print(f"patient_level_split_guaranteed: {patient_level}")
    print(f"image_shape_first: {dataset[0]['image'].shape}")
    print(f"mask_shape_first: {dataset[0]['mask'].shape}")
    print(f"image_min_mean_max: {stats['image_min']:.4f} / {stats['image_mean']:.4f} / {stats['image_max']:.4f}")
    print(f"mask_positive_samples: {stats['positive_samples']} / {len(dataset)}")
    print(f"mask_empty_samples: {stats['empty_samples']} / {len(dataset)}")
    print(f"mask_positive_pixel_ratio_mean: {stats['pos_ratio_mean']:.6f}")
    print(f"mask_positive_pixel_ratio_min_max: {stats['pos_ratio_min']:.6f} / {stats['pos_ratio_max']:.6f}")
    print(f"all_background_predictor_slice_dice: {stats['empty_pred_dice']:.6f}")
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_patients = set(patients[i] for i in train_idx)
        val_patients = set(patients[i] for i in val_idx)
        leakage = sorted(train_patients.intersection(val_patients))
        train_pos = np.mean([stats["sample_pos_ratios"][i] for i in train_idx])
        val_pos = np.mean([stats["sample_pos_ratios"][i] for i in val_idx])
        print(
            f"fold_{fold_idx}: train_samples={len(train_idx)} val_samples={len(val_idx)} "
            f"train_patients={len(train_patients)} val_patients={len(val_patients)} "
            f"patient_leakage={len(leakage)} train_pos_ratio={train_pos:.6f} val_pos_ratio={val_pos:.6f}"
        )

    if args.save_preview:
        out_dir = Path(resolve_path(cfg["output"]["output_dir"])) / "debug_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(min(args.max_preview, len(dataset))):
            item = dataset[idx]
            image = item["image"][0].numpy()
            mask = item["mask"][0].numpy()
            preview = make_preview(image, mask)
            cv2.imwrite(str(out_dir / f"{idx:04d}_{Path(item['image_path']).stem}.png"), preview)
        print(f"preview_dir: {out_dir}")


def collect_dataset_stats(dataset: BreastDM2DDataset):
    image_mins, image_means, image_maxs = [], [], []
    pos_ratios, empty_pred_rows = [], []
    positive_samples = 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        image = item["image"].numpy()
        mask = item["mask"][0].numpy()
        image_mins.append(float(image.min()))
        image_means.append(float(image.mean()))
        image_maxs.append(float(image.max()))
        pos_ratio = float(mask.mean())
        pos_ratios.append(pos_ratio)
        positive_samples += int(pos_ratio > 0)
        empty_pred_rows.append(compute_sample_metrics(np.zeros_like(mask), mask)["dice"])
    return {
        "image_min": float(np.mean(image_mins)),
        "image_mean": float(np.mean(image_means)),
        "image_max": float(np.mean(image_maxs)),
        "positive_samples": positive_samples,
        "empty_samples": len(dataset) - positive_samples,
        "pos_ratio_mean": float(np.mean(pos_ratios)),
        "pos_ratio_min": float(np.min(pos_ratios)),
        "pos_ratio_max": float(np.max(pos_ratios)),
        "empty_pred_dice": float(np.mean(empty_pred_rows)),
        "sample_pos_ratios": pos_ratios,
    }


def make_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = (image * 255.0).clip(0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(rgb)
    red[:, :, 2] = 255
    alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.45
    return (rgb * (1.0 - alpha) + red * alpha).astype(np.uint8)


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
