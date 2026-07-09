import argparse
import re
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
    iter_outputs,
    make_grid,
    normalize_to_uint8,
    normalize_to_uint8_fixed,
    prediction_context_panels,
    resize_map_to_shape,
)


def safe_name(label: str) -> str:
    label = str(label).strip().lower().replace(" ", "_").replace("+", "plus").replace("-", "minus")
    label = re.sub(r"[^0-9a-zA-Z_]+", "_", label)
    return label.strip("_") or "slice"


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
            labels.append("0")
        elif offset < 0:
            labels.append(str(offset))
        else:
            labels.append(f"+{offset}")
    return labels


def center_triplet_indices(num_slices: int):
    if num_slices <= 0:
        return []
    center = num_slices // 2
    return [idx for idx in (center - 1, center, center + 1) if 0 <= idx < num_slices]


def finite_values(maps: np.ndarray) -> np.ndarray:
    values = np.asarray(maps, dtype=np.float32)
    values = values[np.isfinite(values)]
    return values if values.size else np.asarray([0.0], dtype=np.float32)


def resolve_vmax(maps: np.ndarray, explicit_vmax, percentile):
    if explicit_vmax is not None:
        return float(explicit_vmax)
    values = finite_values(maps)
    if percentile is None:
        return float(values.max())
    percentile = float(np.clip(percentile, 50.0, 100.0))
    return float(np.percentile(values, percentile))


def boost_attention_colormap(norm: np.ndarray, gamma: float, start: float) -> np.ndarray:
    norm = np.clip(norm.astype(np.float32), 0.0, 1.0)
    start = float(np.clip(start, 0.0, 0.95))
    gamma = max(float(gamma), 1e-6)
    boosted = norm.copy()
    active = norm > start
    if np.any(active):
        local = (norm[active] - start) / max(1.0 - start, 1e-6)
        boosted[active] = start + (1.0 - start) * np.power(local, gamma)
    return np.clip(boosted, 0.0, 1.0)


def attention_overlay_boosted(base: np.ndarray, display_map: np.ndarray, vmin: float, vmax: float, alpha: float, gamma: float, boost_start: float) -> np.ndarray:
    norm = normalize_to_uint8_fixed(display_map, vmin=vmin, vmax=vmax).astype(np.float32) / 255.0
    norm = boost_attention_colormap(norm, gamma=gamma, start=boost_start)
    heatmap = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(gray_to_bgr(base), 1.0 - alpha, heatmap, alpha, 0)


def save_attention_group(output_root: Path, stem: str, base: np.ndarray, maps: np.ndarray, prefix: str, sample, args):
    if maps.size == 0:
        return False

    labels = slice_labels(maps.shape[0])
    indices = center_triplet_indices(maps.shape[0])
    vmin = float(args.slice_vmin)
    vmax = resolve_vmax(maps[indices] if indices else maps, args.slice_vmax, args.slice_vmax_percentile)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    panels = [add_title(gray_to_bgr(base), "center pre")]
    panels.extend(prediction_context_panels(sample, base))
    for index in indices:
        label = labels[index]
        att = maps[index]
        display_map = resize_map_to_shape(att, base.shape[:2])
        overlay = attention_overlay_boosted(
            base,
            display_map,
            vmin=vmin,
            vmax=vmax,
            alpha=float(np.clip(args.slice_overlay_alpha, 0.0, 1.0)),
            gamma=args.slice_colormap_gamma,
            boost_start=args.slice_boost_start,
        )
        panels.append(add_title(overlay, f"{prefix} {label} mean={float(att.mean()):.3f}"))
        name = safe_name(label)
        cv2.imwrite(str(output_root / f"{stem}_{prefix}_{name}_attention.png"), overlay)
        cv2.imwrite(
            str(output_root / f"{stem}_{prefix}_{name}_attention_raw.png"),
            normalize_to_uint8_fixed(display_map, vmin=vmin, vmax=vmax),
        )
    cv2.imwrite(str(output_root / f"{stem}_{prefix}_grid.png"), make_grid(panels, cols=4))
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize slice attention maps with GT and prediction context.")
    add_common_args(parser, "slice_attention")
    parser.add_argument("--slice_vmin", type=float, default=0.0, help="Lower bound for slice-attention colormap. Default: 0.")
    parser.add_argument("--slice_vmax", type=float, default=None, help="Upper bound for slice-attention colormap. Default: per-sample max attention.")
    parser.add_argument(
        "--slice_vmax_percentile",
        type=float,
        default=None,
        help="Optional percentile used as slice_vmax when --slice_vmax is not set. Lower values make hotspots redder.",
    )
    parser.add_argument(
        "--slice_overlay_alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity. Default matches the original grid style: 0.45.",
    )
    parser.add_argument(
        "--slice_colormap_gamma",
        type=float,
        default=0.55,
        help="Nonlinear boost for mid/high attention colors. Lower values make yellow regions redder. Default: 0.55.",
    )
    parser.add_argument(
        "--slice_boost_start",
        type=float,
        default=0.25,
        help="Normalized attention value where red-boost starts. Lower values affect more area. Default: 0.25.",
    )
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
        wrote_image = save_attention_group(output_root, stem, base, image_slice_attn, "image_slice", sample, args)
        wrote_kinetic = save_attention_group(output_root, stem, base, kinetic_slice_attn, "kinetic_slice", sample, args)
        if wrote_image or wrote_kinetic:
            saved += 1
    print(f"Saved slice attention visualizations for {saved} samples")


if __name__ == "__main__":
    main()
