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
    gray_to_bgr,
    iter_outputs,
    normalize_to_uint8,
    resize_map_to_shape,
)
from visual.paper_style import (
    PLASMA_COLORMAP,
    compose_case_matrix,
    kinetic_display_indices,
    phase_panel,
    prior_heatmap,
    tensor_to_numpy,
)

COLUMN_LABELS = ["Pre", "Post-1", "Post-2", "Post-3", "Subtraction", "Pseudo-kinetic\nPrior"]


def kinetic_prior(maps: np.ndarray) -> np.ndarray:
    maps = np.nan_to_num(np.asarray(maps, dtype=np.float32))
    if maps.ndim == 2:
        return maps
    if maps.ndim > 3:
        maps = maps.reshape((-1, maps.shape[-2], maps.shape[-1]))
    return np.sqrt(np.mean(np.square(maps), axis=0))


def build_kinetic_row(sample, kinetic: torch.Tensor, stem: str, output_root: Path):
    if kinetic is None or not torch.is_tensor(kinetic):
        return None
    kinetic = kinetic.detach().float()
    if kinetic.ndim != 4:
        return None

    image = tensor_to_numpy(sample["image"])
    center_slice = image.shape[0] // 2
    count = image.shape[1]
    phase_indices = kinetic_display_indices(sample["config"], count)
    panels = [phase_panel(image, center_slice, index) for index in phase_indices]

    maps = kinetic[center_slice].numpy()
    prior = resize_map_to_shape(kinetic_prior(maps), image.shape[-2:])
    prior_panel = prior_heatmap(prior)
    panels.append(prior_panel)

    cv2.imwrite(str(output_root / f"{stem}_pseudo_kinetic_prior.png"), prior_panel)
    cv2.imwrite(str(output_root / f"{stem}_pseudo_kinetic_prior_gray.png"), normalize_to_uint8(prior))
    for label, phase_index, panel in zip(COLUMN_LABELS[:-1], phase_indices, panels[:-1]):
        cv2.imwrite(str(output_root / f"{stem}_{label.lower().replace('-', '').replace(' ', '_')}_{phase_index}.png"), panel)
    return panels


def main():
    parser = argparse.ArgumentParser(description="Visualize pseudo-kinetic enhancement priors in a paper-style case matrix.")
    add_common_args(parser, "kinetic_maps")
    parser.add_argument("--panel_size", type=int, default=150, help="Square panel size in pixels for the summary figure.")
    args = parser.parse_args()

    rows = []
    last_output_root = None
    for sample in iter_outputs(args, "Pseudo-kinetic enhancement prior visualization"):
        output_root = Path(sample["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        last_output_root = output_root
        stem = Path(sample["name"]).stem
        row = build_kinetic_row(sample, sample["output"].get("kinetic_maps"), stem, output_root)
        if row is not None:
            rows.append(row)

    if last_output_root is not None and rows:
        compose_case_matrix(
            "Pseudo-kinetic Enhancement Priors",
            rows,
            COLUMN_LABELS,
            last_output_root / "pseudo_kinetic_enhancement_priors.png",
            panel_size=(int(args.panel_size), int(args.panel_size)),
            colorbar={
                "colormap": PLASMA_COLORMAP,
                "label": "Enhancement Amplitude",
                "tick_labels": [(1.0, "High"), (0.0, "Low")],
            },
        )
    print(f"Saved pseudo-kinetic enhancement prior summary for {len(rows)} samples")


if __name__ == "__main__":
    main()
