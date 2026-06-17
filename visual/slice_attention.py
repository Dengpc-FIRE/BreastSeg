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
)


def reduce_slice_attention(attn) -> np.ndarray:
    """Convert a slice-attention tensor to [K,H,W]."""
    if attn is None or not torch.is_tensor(attn):
        return np.empty((0, 0, 0), dtype=np.float32)
    tensor = attn.detach().float()
    if tensor.ndim == 5:
        # [K,T,1,H,W] or [K,T,C,H,W] -> [K,H,W]
        tensor = tensor.mean(dim=(1, 2))
    elif tensor.ndim == 4:
        # [K,1,H,W] or [K,T,H,W] -> [K,H,W]
        tensor = tensor.mean(dim=1)
    elif tensor.ndim == 3:
        pass
    else:
        return np.empty((0, 0, 0), dtype=np.float32)
    return tensor.numpy()


def save_attention_group(output_root: Path, stem: str, base: np.ndarray, maps: np.ndarray, prefix: str):
    if maps.size == 0:
        return
    labels = [f"z-{maps.shape[0] // 2 - i}" if i < maps.shape[0] // 2 else ("z" if i == maps.shape[0] // 2 else f"z+{i - maps.shape[0] // 2}") for i in range(maps.shape[0])]
    panels = [add_title(gray_to_bgr(base), "center pre")]
    for index, att in enumerate(maps):
        hm = heatmap(att)
        overlay = cv2.addWeighted(gray_to_bgr(base), 0.55, hm, 0.45, 0)
        panels.append(add_title(overlay, f"{prefix} {labels[index]} mean={att.mean():.3f}"))
        cv2.imwrite(
            str(output_root / f"{stem}_{prefix}_{index}.png"),
            normalize_to_uint8(att),
        )
    cv2.imwrite(str(output_root / f"{stem}_{prefix}_grid.png"), make_grid(panels, cols=4))


def main():
    parser = argparse.ArgumentParser(description="Visualize slice attention maps.")
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
        save_attention_group(output_root, stem, base, image_slice_attn, "image_slice")
        save_attention_group(output_root, stem, base, kinetic_slice_attn, "kinetic_slice")
        saved += 1
    print(f"Saved slice attention visualizations for {saved} samples")


if __name__ == "__main__":
    main()
