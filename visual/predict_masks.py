import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.common import (
    add_common_args,
    add_title,
    center_pre,
    gray_to_bgr,
    iter_outputs,
    normalize_to_uint8,
    overlay_mask,
    write_csv,
)


def binary_metrics(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    return {
        "dice": 1.0 if dice_den == 0 else 2.0 * tp / dice_den,
        "iou": 1.0 if iou_den == 0 else tp / iou_den,
        "sensitivity": 1.0 if tp + fn == 0 else tp / (tp + fn),
        "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser(description="Save predicted tumor masks and overlays.")
    add_common_args(parser, "predict_masks")
    parser.set_defaults(max_samples=1_000_000)
    args = parser.parse_args()

    rows = []
    for sample in iter_outputs(args, "Predict mask visualization"):
        output_root = Path(sample["output_root"])
        for directory in ("masks", "probabilities", "overlays", "panels"):
            (output_root / directory).mkdir(parents=True, exist_ok=True)

        stem = Path(sample["name"]).stem
        base = normalize_to_uint8(center_pre(sample["image"]))
        gt = (sample["mask"][0].numpy() >= 0.5).astype(np.uint8)
        prob = sample["probability"][0].numpy()
        pred = (prob >= sample["threshold"]).astype(np.uint8)
        breast = (sample["breast_mask"][0].numpy() >= 0.5).astype(np.uint8)

        metrics = binary_metrics(pred, gt)
        rows.append({"name": sample["name"], **metrics})

        cv2.imwrite(str(output_root / "masks" / f"{stem}.png"), pred * 255)
        cv2.imwrite(
            str(output_root / "probabilities" / f"{stem}.png"),
            np.clip(prob * 255.0, 0, 255).astype(np.uint8),
        )

        gt_overlay = overlay_mask(base, gt, color=(0, 255, 0), alpha=0.50)
        pred_overlay = overlay_mask(base, pred, color=(0, 0, 255), alpha=0.50)
        breast_overlay = overlay_mask(base, breast, color=(255, 0, 0), alpha=0.25)
        merged = gray_to_bgr(base)
        merged[gt.astype(bool)] = (0, 255, 0)
        merged[pred.astype(bool)] = (0, 0, 255)
        overlap = gt.astype(bool) & pred.astype(bool)
        merged[overlap] = (0, 255, 255)

        panels = [
            add_title(gray_to_bgr(base), "center pre"),
            add_title(gray_to_bgr(gt * 255), "GT mask"),
            add_title(gray_to_bgr(pred * 255), f"pred mask t={sample['threshold']:g}"),
            add_title(gt_overlay, "GT overlay"),
            add_title(pred_overlay, "prediction overlay"),
            add_title(breast_overlay, "whole-breast ROI"),
            add_title(merged, f"GT green / Pred red / Dice {metrics['dice']:.3f}"),
        ]
        panel = np.hstack(panels)
        cv2.imwrite(str(output_root / "panels" / f"{stem}.png"), panel)
        cv2.imwrite(str(output_root / "overlays" / f"{stem}.png"), merged)

    if rows:
        write_csv(
            Path(sample["output_root"]) / "metrics.csv",
            rows,
            ["name", "dice", "iou", "sensitivity", "precision", "tp", "fp", "fn", "tn"],
        )
        print(f"Saved {len(rows)} mask visualizations to {sample['output_root']}")


if __name__ == "__main__":
    main()
