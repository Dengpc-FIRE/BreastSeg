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
    iter_outputs,
    resize_map_to_shape,
)
from visual.paper_style import (
    compose_case_matrix,
    overlay_attention,
    save_fixed_map,
    slice_window_indices,
    tensor_to_numpy,
)


def reduce_slice_attention(attn) -> np.ndarray:
    """Convert a slice-attention tensor to [K,H,W] or [K,1,1]."""
    if attn is None or not torch.is_tensor(attn):
        return np.empty((0, 0, 0), dtype=np.float32)
    tensor = attn.detach().float()
    if tensor.ndim == 5:
        # [K,T,1,H,W] or [K,T,C,H,W] -> [K,H,W].
        tensor = tensor.mean(dim=(1, 2))
    elif tensor.ndim == 4:
        # [K,1,H,W] or [K,T,H,W] -> [K,H,W].
        tensor = tensor.mean(dim=1)
    elif tensor.ndim == 3:
        pass
    else:
        return np.empty((0, 0, 0), dtype=np.float32)
    return tensor.numpy()


def build_attention_row(sample, maps: np.ndarray, stem: str, output_root: Path, prefix: str, vmin: float, vmax: float):
    if maps.size == 0:
        return None
    image = tensor_to_numpy(sample["image"])
    indices = slice_window_indices(maps.shape[0])
    if len(indices) != 3:
        return None

    panels = []
    for label, map_index in zip(["-1", "0", "+1"], indices):
        image_index = int(np.clip(map_index, 0, image.shape[0] - 1))
        base = image[image_index, 0]
        display_map = resize_map_to_shape(maps[map_index], base.shape[:2])
        panels.append(overlay_attention(base, display_map, vmin=vmin, vmax=vmax))
        save_fixed_map(output_root / f"{stem}_{prefix}_{label}_attention.png", display_map, vmin=vmin, vmax=vmax)
    return panels


def main():
    parser = argparse.ArgumentParser(description="Visualize slice attention maps in a paper-style case matrix.")
    add_common_args(parser, "slice_attention")
    parser.add_argument("--slice_vmin", type=float, default=0.0)
    parser.add_argument("--slice_vmax", type=float, default=1.0)
    parser.add_argument("--panel_size", type=int, default=150, help="Square panel size in pixels for the summary figure.")
    args = parser.parse_args()

    image_rows = []
    kinetic_rows = []
    last_output_root = None
    for sample in iter_outputs(args, "Slice attention visualization"):
        output_root = Path(sample["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        last_output_root = output_root
        stem = Path(sample["name"]).stem
        maps = sample["output"].get("slice_attention_maps", [])
        image_slice_attn = reduce_slice_attention(maps[0] if maps else None)
        kinetic_slice_attn = reduce_slice_attention(maps[1] if len(maps) > 1 else None)

        image_row = build_attention_row(sample, image_slice_attn, stem, output_root, "image_slice", args.slice_vmin, args.slice_vmax)
        if image_row is not None:
            image_rows.append(image_row)
        kinetic_row = build_attention_row(sample, kinetic_slice_attn, stem, output_root, "kinetic_slice", args.slice_vmin, args.slice_vmax)
        if kinetic_row is not None:
            kinetic_rows.append(kinetic_row)

    colorbar = {
        "colormap": cv2.COLORMAP_JET,
        "label": "Attention Weight",
        "tick_labels": [(1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")],
    }
    panel_size = (int(args.panel_size), int(args.panel_size))
    if last_output_root is not None and image_rows:
        compose_case_matrix(
            "Slice Attention Visualization",
            image_rows,
            ["-1", "0", "+1"],
            last_output_root / "slice_attention_visualization.png",
            panel_size=panel_size,
            colorbar=colorbar,
        )
    if last_output_root is not None and kinetic_rows:
        compose_case_matrix(
            "Kinetic Slice Attention Visualization",
            kinetic_rows,
            ["-1", "0", "+1"],
            last_output_root / "kinetic_slice_attention_visualization.png",
            panel_size=panel_size,
            colorbar=colorbar,
        )
    print(f"Saved slice attention summary for {len(image_rows)} samples")


if __name__ == "__main__":
    main()
