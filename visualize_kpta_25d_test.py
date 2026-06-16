import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.train_config import (
    build_model_from_config,
    checkpoint_name_from_config,
    load_config,
    resolve_config_path,
)
from inference.whole_breast_constraint import build_whole_breast_constraint
from visualize_kpta_25d_val import (
    KPTA25DSegmentationDataset,
    add_title,
    binary_metrics,
    color_overlay,
    extract_state_dict,
    normalize_for_display,
    validate_full_config,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run scheme D-full best model on the test set and save 2D masks."
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
        help="Default: train.test_path from the YAML config",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <train.output_path>/test_visualization",
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

    split_path = Path(args.split_path or train_cfg.get("test_path", ""))
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
            / "test_visualization"
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
    print(f"Test set: {split_path} ({len(dataset)} samples)")
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
        for images, masks, names in tqdm(loader, desc="Visualizing test set"):
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
                probability_u8 = np.clip(
                    probability * 255.0, 0, 255
                ).astype(np.uint8)

                center_slice = images_cpu[index].shape[0] // 2
                base = normalize_for_display(images_cpu[index, center_slice, 0])
                original_panel = add_title(
                    cv2.cvtColor(base, cv2.COLOR_GRAY2BGR),
                    "Center pre-contrast",
                )
                gt_panel = add_title(
                    cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR),
                    "Ground truth",
                )
                pred_panel = add_title(
                    cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR),
                    f"Prediction (t={args.threshold:g})",
                )
                overlay_panel = add_title(
                    color_overlay(base, gt, pred),
                    f"Overlay D={metrics['dice']:.3f} GT=green Pred=red",
                )
                visualization = np.hstack(
                    [original_panel, gt_panel, pred_panel, overlay_panel]
                )

                cv2.imwrite(str(mask_dir / f"{stem}.png"), pred_u8)
                cv2.imwrite(
                    str(probability_dir / f"{stem}.png"),
                    probability_u8,
                )
                cv2.imwrite(
                    str(visualization_dir / f"{stem}.png"),
                    visualization,
                )

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
