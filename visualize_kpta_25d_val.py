import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_config import (
    build_model_from_config,
    checkpoint_name_from_config,
    load_config,
    resolve_config_path,
)
from inference.whole_breast_constraint import build_whole_breast_constraint


class KPTA25DSegmentationDataset(Dataset):
    """Load processed SA-KPTA-Net samples with shape [K,T,H,W]."""

    def __init__(self, split_path: str, input_phase_indices=None):
        self.split_path = Path(split_path)
        self.data_dir = self.split_path / "data"
        self.gt_dir = self.split_path / "GT"
        self.input_phase_indices = (
            [int(index) for index in input_phase_indices]
            if input_phase_indices is not None
            else None
        )
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")
        self.files = sorted(self.data_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy samples found in: {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_path = self.files[index]
        image = np.load(data_path).astype(np.float32)
        if image.ndim != 4:
            raise ValueError(
                f"Expected [K,T,H,W] input, got {image.shape} for {data_path.name}"
            )
        if self.input_phase_indices is not None:
            image = image[:, self.input_phase_indices]

        gt_path = self.gt_dir / f"{data_path.stem}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"GT mask not found or unreadable: {gt_path}")
        height, width = image.shape[-2:]
        if gt.shape != (height, width):
            gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)
        gt = (gt > 127).astype(np.float32)

        return (
            torch.from_numpy(image),
            torch.from_numpy(gt).unsqueeze(0),
            data_path.name,
        )


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32))
    low, high = np.percentile(image, (1.0, 99.0))
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def add_title(panel: np.ndarray, title: str) -> np.ndarray:
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def color_overlay(base: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Green=GT only, red=prediction only, yellow=overlap."""
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    gt_only = (gt > 0) & (pred == 0)
    pred_only = (pred > 0) & (gt == 0)
    overlap = (gt > 0) & (pred > 0)
    colors = overlay.copy()
    colors[gt_only] = (0, 255, 0)
    colors[pred_only] = (0, 0, 255)
    colors[overlap] = (0, 255, 255)
    active = gt_only | pred_only | overlap
    overlay[active] = (
        0.45 * overlay[active].astype(np.float32)
        + 0.55 * colors[active].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    """Return the 2D binary boundary pixels of a mask."""
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask & ~eroded


def hd95_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute symmetric 95th percentile Hausdorff distance in pixels.

    The processed BreastDM samples do not carry physical pixel spacing, so this
    value is reported in resized-image pixels rather than millimeters.
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    if not pred_bool.any() and not gt_bool.any():
        return 0.0
    if not pred_bool.any() or not gt_bool.any():
        height, width = pred_bool.shape
        return float(np.hypot(height, width))

    pred_surface = _binary_surface(pred_bool)
    gt_surface = _binary_surface(gt_bool)
    if not pred_surface.any() or not gt_surface.any():
        height, width = pred_bool.shape
        return float(np.hypot(height, width))

    # cv2.distanceTransform computes distance to the nearest zero pixel.
    # Therefore surface pixels of the opposite mask are encoded as zero.
    dist_to_gt = cv2.distanceTransform(
        (~gt_surface).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    dist_to_pred = cv2.distanceTransform(
        (~pred_surface).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    distances = np.concatenate(
        [dist_to_gt[pred_surface], dist_to_pred[gt_surface]]
    )
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def binary_metrics(pred: np.ndarray, gt: np.ndarray):
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())

    dice_denom = 2 * tp + fp + fn
    iou_denom = tp + fp + fn
    dice = 1.0 if dice_denom == 0 else (2.0 * tp) / dice_denom
    iou = 1.0 if iou_denom == 0 else tp / iou_denom
    sensitivity = 1.0 if tp + fn == 0 else tp / (tp + fn)
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "dice": dice,
        "iou": iou,
        "hd95": hd95_2d(pred_bool, gt_bool),
        "sensitivity": sensitivity,
        "precision": precision,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def validate_full_config(config):
    model_name = config.get("model", {}).get("name")
    if model_name not in {"kpta_25d_net", "kpta25dnet", "KPTA25DNet", "sa_kpta_net"}:
        raise ValueError(f"Expected a KPTA-2.5D config, got model.name={model_name!r}")
    disabled = [
        key
        for key, value in config.get("ablation", {}).items()
        if key.startswith("disable_") and bool(value)
    ]
    if disabled:
        raise ValueError(
            "The selected config is not scheme D-full; enabled ablations: "
            + ", ".join(disabled)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run scheme D-full best model on the validation set and save 2D masks."
    )
    parser.add_argument("--config", default="configs/kpta_25d_net.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Default: <train.output_path>/best_model_<config-name>.pth. "
            "An explicit legacy best_model.pth path is still supported."
        ),
    )
    parser.add_argument(
        "--split_path",
        default=None,
        help="Default: train.val_path from the YAML config",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <train.output_path>/validation_visualization",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    validate_full_config(config)
    train_cfg = config.get("train", {})

    split_path = Path(args.split_path or train_cfg.get("val_path", ""))
    checkpoint_path = Path(
        args.checkpoint
        or (
            Path(train_cfg.get("output_path", "./results_kpta_25d_net"))
            / checkpoint_name_from_config(config_path)
        )
    )
    output_dir = Path(
        args.output_dir
        or (
            Path(train_cfg.get("output_path", "./results_kpta_25d_net"))
            / "validation_visualization"
        )
    )
    mask_dir = output_dir / "masks"
    probability_dir = output_dir / "probabilities"
    visualization_dir = output_dir / "visualizations"
    for directory in (mask_dir, probability_dir, visualization_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    input_phase_indices = config.get("dataset", {}).get("input_phase_indices")
    dataset = KPTA25DSegmentationDataset(
        str(split_path),
        input_phase_indices=input_phase_indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    whole_breast_constraint = build_whole_breast_constraint(
        config,
        device=device,
        output_path=Path(train_cfg.get("output_path", "./results_kpta_25d_net")),
    )

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation set: {split_path} ({len(dataset)} samples)")
    print(
        "Input phase indices: "
        f"{input_phase_indices if input_phase_indices is not None else 'all'}"
    )
    print(f"Output: {output_dir}")
    print(
        "Whole-breast inference constraint: "
        f"{'enabled' if whole_breast_constraint is not None else 'disabled'}"
    )

    rows = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for images, masks, names in tqdm(loader, desc="Visualizing validation set"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(images, return_dict=True)
                logits = output["seg_logits"] if isinstance(output, dict) else output
                probabilities = torch.sigmoid(logits)
            if whole_breast_constraint is not None:
                probabilities, _ = (
                    whole_breast_constraint.constrain_probabilities(
                        probabilities,
                        names,
                        dataset,
                    )
                )

            probabilities = probabilities[:, 0].float().cpu().numpy()
            masks = masks[:, 0].numpy()
            images_cpu = images.float().cpu().numpy()

            for index, name in enumerate(names):
                stem = Path(name).stem
                probability = probabilities[index]
                pred = (probability >= args.threshold).astype(np.uint8)
                gt = (masks[index] >= 0.5).astype(np.uint8)
                metrics = binary_metrics(pred, gt)
                rows.append({"name": name, **metrics})

                pred_u8 = pred * 255
                gt_u8 = gt * 255
                probability_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)

                # Display the pre-contrast phase from the center slice.
                center_slice = images_cpu[index].shape[0] // 2
                base = normalize_for_display(images_cpu[index, center_slice, 0])
                original_panel = add_title(
                    cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), "Center pre-contrast"
                )
                gt_panel = add_title(
                    cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR), "Ground truth"
                )
                pred_panel = add_title(
                    cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR),
                    f"Prediction (t={args.threshold:g})",
                )
                overlay_panel = add_title(
                    color_overlay(base, gt, pred),
                    (
                        f"Overlay D={metrics['dice']:.3f} "
                        "GT=green Pred=red"
                    ),
                )
                visualization = np.hstack(
                    [original_panel, gt_panel, pred_panel, overlay_panel]
                )

                cv2.imwrite(str(mask_dir / f"{stem}.png"), pred_u8)
                cv2.imwrite(str(probability_dir / f"{stem}.png"), probability_u8)
                cv2.imwrite(str(visualization_dir / f"{stem}.png"), visualization)

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "name",
        "dice",
        "iou",
        "hd95",
        "sensitivity",
        "precision",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = ("dice", "iou", "hd95", "sensitivity", "precision", "accuracy")
    means = {
        key: float(np.mean([row[key] for row in rows])) if rows else 0.0
        for key in metric_names
    }
    summary = (
        f"samples: {len(rows)}\n"
        f"threshold: {args.threshold:g}\n"
        f"mean_dice: {means['dice']:.6f}\n"
        f"mean_iou: {means['iou']:.6f}\n"
        f"mean_hd95: {means['hd95']:.6f}\n"
        f"mean_sensitivity: {means['sensitivity']:.6f}\n"
        f"mean_precision: {means['precision']:.6f}\n"
        f"mean_accuracy: {means['accuracy']:.6f}\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"2D masks saved to: {mask_dir}")
    print(f"Visualizations saved to: {visualization_dir}")


if __name__ == "__main__":
    main()
