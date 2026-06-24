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
    heatmap_overlay,
    iter_outputs,
    make_grid,
    normalize_to_uint8,
    prediction_context_panels,
    resize_map_to_shape,
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


def slice_labels(num_slices: int):
    center = num_slices // 2
    labels = []
    for index in range(num_slices):
        offset = index - center
        if offset == 0:
            labels.append("z")
        elif offset < 0:
            labels.append(f"z{offset}")
        else:
            labels.append(f"z+{offset}")
    return labels


def save_attention_group(output_root: Path, stem: str, base: np.ndarray, maps: np.ndarray, prefix: str, sample):
    if maps.size == 0:
        return False
    labels = slice_labels(maps.shape[0])
    panels = [add_title(gray_to_bgr(base), "center pre")]
    panels.extend(prediction_context_panels(sample, base))
    for index, att in enumerate(maps):
        display_map = resize_map_to_shape(att, base.shape[:2])
        overlay = heatmap_overlay(base, display_map)
        panels.append(add_title(overlay, f"{prefix} {labels[index]} mean={float(att.mean()):.3f}"))
        cv2.imwrite(
            str(output_root / f"{stem}_{prefix}_{labels[index]}_attention.png"),
            normalize_to_uint8(display_map),
        )
    cv2.imwrite(str(output_root / f"{stem}_{prefix}_grid.png"), make_grid(panels, cols=4))
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize slice attention maps with GT and prediction context.")
    add_common_args(parser, "slice_attention")
    args = parser.parse_args()

    saved = 0
    for sample in iter_outputs(args, "Slice attention visualization"):
        output_root = Path(sample["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        stem = Path(sample["name"]).stem
        base = normalize_to_uint8(center_pre(sample["image"]))
        maps = sample["output"].get("slice_attention_maps", [])
        image_slice_attn = reduce_slice_attention(maps[0] if maps else None)
        kinetic_slice_attn = reduce_slice_attention(maps[1] if len(maps) > 1 else None)
        wrote_image = save_attention_group(output_root, stem, base, image_slice_attn, "image_slice", sample)
        wrote_kinetic = save_attention_group(output_root, stem, base, kinetic_slice_attn, "kinetic_slice", sample)
        if wrote_image or wrote_kinetic:
            saved += 1
    print(f"Saved slice attention visualizations for {saved} samples")


if __name__ == "__main__":
    main()
