import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.common import (
    add_common_args,
    add_title,
    center_pre,
    gray_to_bgr,
    heatmap,
    iter_outputs,
    make_grid,
    normalize_to_uint8,
    overlay_mask,
)


def sigmoid_map(tensor) -> np.ndarray:
    if tensor is None or not torch.is_tensor(tensor):
        return np.empty((0, 0), dtype=np.float32)
    value = torch.sigmoid(tensor.detach().float())
    if value.ndim == 3:
        value = value[0]
    return value.numpy()


def plain_map(tensor) -> np.ndarray:
    if tensor is None or not torch.is_tensor(tensor):
        return np.empty((0, 0), dtype=np.float32)
    value = tensor.detach().float()
    if value.ndim == 3:
        value = value[0]
    return value.numpy()


def main():
    parser = argparse.ArgumentParser(description="Visualize boundary and uncertainty outputs.")
    add_common_args(parser, "uncertainty_boundary")
    args = parser.parse_args()

    saved = 0
    for sample in iter_outputs(args, "Boundary/uncertainty visualization"):
        output_root = Path(sample["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        stem = Path(sample["name"]).stem
        base = normalize_to_uint8(center_pre(sample["image"]))
        gt = (sample["mask"][0].numpy() >= 0.5).astype(np.uint8)
        prob = sample["probability"][0].numpy()
        pred = (prob >= sample["threshold"]).astype(np.uint8)

        coarse = sigmoid_map(sample["output"].get("coarse_logits"))
        boundary = sigmoid_map(sample["output"].get("boundary_logits"))
        uncertainty = plain_map(sample["output"].get("uncertainty_map"))
        if uncertainty.size == 0:
            uncertainty = sigmoid_map(sample["output"].get("uncertainty_logits"))

        panels = [
            add_title(gray_to_bgr(base), "center pre"),
            add_title(gray_to_bgr(gt * 255), "GT mask"),
            add_title(heatmap(prob), "tumor probability"),
            add_title(gray_to_bgr(pred * 255), "prediction"),
            add_title(overlay_mask(base, pred, color=(0, 0, 255)), "prediction overlay"),
        ]
        if coarse.size:
            panels.append(add_title(heatmap(coarse), "coarse probability"))
            cv2.imwrite(str(output_root / f"{stem}_coarse_probability.png"), normalize_to_uint8(coarse))
        if uncertainty.size:
            panels.append(add_title(heatmap(uncertainty), "uncertainty"))
            cv2.imwrite(str(output_root / f"{stem}_uncertainty.png"), normalize_to_uint8(uncertainty))
        if boundary.size:
            panels.append(add_title(heatmap(boundary), "boundary probability"))
            cv2.imwrite(str(output_root / f"{stem}_boundary_probability.png"), normalize_to_uint8(boundary))
            boundary_binary = boundary >= 0.5
            panels.append(add_title(overlay_mask(base, boundary_binary, color=(255, 0, 255), alpha=0.45), "boundary overlay"))

        cv2.imwrite(str(output_root / f"{stem}_uncertainty_boundary_grid.png"), make_grid(panels, cols=4))
        saved += 1
    print(f"Saved boundary/uncertainty visualizations for {saved} samples")


if __name__ == "__main__":
    main()
