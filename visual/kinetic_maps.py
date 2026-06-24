import argparse
import sys
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.common import (
    add_common_args,
    add_title,
    center_pre,
    gray_to_bgr,
    heatmap_overlay,
    iter_outputs,
    kinetic_labels,
    make_grid,
    normalize_to_uint8,
    prediction_context_panels,
    resize_map_to_shape,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize pseudo-kinetic maps with GT and prediction context.")
    add_common_args(parser, "kinetic_maps")
    args = parser.parse_args()

    saved = 0
    for sample in iter_outputs(args, "Kinetic map visualization"):
        output_root = Path(sample["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        stem = Path(sample["name"]).stem
        base = normalize_to_uint8(center_pre(sample["image"]))
        kinetic = sample["output"].get("kinetic_maps")
        if kinetic is None or not torch.is_tensor(kinetic):
            continue
        kinetic = kinetic.detach().float()
        if kinetic.ndim != 4:
            continue

        center = kinetic.shape[0] // 2
        maps = kinetic[center].numpy()  # [M,H,W], center slice kinetic maps.
        labels = kinetic_labels(sample["config"], maps.shape[0])
        panels = [add_title(gray_to_bgr(base), "center pre")]
        panels.extend(prediction_context_panels(sample, base))

        for index, kinetic_map in enumerate(maps):
            display_map = resize_map_to_shape(kinetic_map, base.shape[:2])
            overlay = heatmap_overlay(base, display_map)
            panels.append(add_title(overlay, labels[index]))
            cv2.imwrite(
                str(output_root / f"{stem}_{index:02d}_{labels[index]}.png"),
                normalize_to_uint8(display_map),
            )
        cv2.imwrite(str(output_root / f"{stem}_kinetic_grid.png"), make_grid(panels, cols=4))
        saved += 1
    print(f"Saved kinetic map visualizations for {saved} samples")


if __name__ == "__main__":
    main()
