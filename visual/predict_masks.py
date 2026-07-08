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
    center_pre,
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
    parser = argparse.ArgumentParser(description="Save predicted tumor masks and red segmentation overlays.")
    add_common_args(parser, "predict_masks")
    parser.set_defaults(max_samples=1_000_000)
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.55,
        help="Transparency for the red predicted segmentation overlay. Default: 0.55.",
    )
    args = parser.parse_args()

    rows = []
    last_output_root = None
    for sample in iter_outputs(args, "Predict mask visualization"):
        output_root = Path(sample["output_root"])
        last_output_root = output_root
        for directory in ("masks", "probabilities", "overlays", "panels"):
            (output_root / directory).mkdir(parents=True, exist_ok=True)

        stem = Path(sample["name"]).stem
        base = normalize_to_uint8(center_pre(sample["image"]))
        gt = (sample["mask"][0].numpy() >= 0.5).astype(np.uint8)
        prob = sample["probability"][0].numpy()
        pred = (prob >= sample["threshold"]).astype(np.uint8)

        metrics = binary_metrics(pred, gt)
        rows.append({"name": sample["name"], **metrics})

        cv2.imwrite(str(output_root / "masks" / f"{stem}.png"), pred * 255)
        cv2.imwrite(
            str(output_root / "probabilities" / f"{stem}.png"),
            np.clip(prob * 255.0, 0, 255).astype(np.uint8),
        )

        pred_overlay = overlay_mask(
            base,
            pred,
            color=(0, 0, 255),
            alpha=float(np.clip(args.overlay_alpha, 0.0, 1.0)),
        )
        cv2.imwrite(str(output_root / "overlays" / f"{stem}.png"), pred_overlay)
        cv2.imwrite(str(output_root / "panels" / f"{stem}.png"), pred_overlay)

    if rows and last_output_root is not None:
        write_csv(
            last_output_root / "metrics.csv",
            rows,
            ["name", "dice", "iou", "sensitivity", "precision", "tp", "fp", "fn", "tn"],
        )
        print(f"Saved {len(rows)} red segmentation overlays to {last_output_root}")


if __name__ == "__main__":
    main()
