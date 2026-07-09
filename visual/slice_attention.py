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


def display_slice_label(offset: int) -> str:
    if offset == 0:
        return "z"
    if offset < 0:
        return f"z{offset}"
    return f"z+{offset}"


def slice_file_suffix(offset: int) -> str:
    if offset == 0:
        return "z"
    if offset < 0:
        return f"z_minus{abs(offset)}"
    return f"z_plus{offset}"


def reduce_slice_attention(attn, reduce_mode: str = "mean", phase_index: int | None = None) -> np.ndarray:
    """Convert a slice-attention tensor to [K,H,W] or [K,1,1]."""
    if attn is None or not torch.is_tensor(attn):
        return np.empty((0, 0, 0), dtype=np.float32)
    tensor = attn.detach().float()
    reduce_mode = str(reduce_mode).lower()
    if tensor.ndim == 5:
        # [K,T,1,H,W] or [K,T,C,H,W] -> [K,H,W].
        if reduce_mode == "phase":
            phase_index = tensor.shape[1] // 2 if phase_index is None else int(phase_index)
            phase_index = int(np.clip(phase_index, 0, tensor.shape[1] - 1))
            tensor = tensor[:, phase_index].mean(dim=1)
        elif reduce_mode == "max":
            tensor = tensor.mean(dim=2).amax(dim=1)
        else:
            tensor = tensor.mean(dim=(1, 2))
    elif tensor.ndim == 4:
        # [K,1,H,W] or [K,T,H,W] -> [K,H,W].
        if reduce_mode == "phase" and tensor.shape[1] > 1:
            phase_index = tensor.shape[1] // 2 if phase_index is None else int(phase_index)
            phase_index = int(np.clip(phase_index, 0, tensor.shape[1] - 1))
            tensor = tensor[:, phase_index]
        elif reduce_mode == "max" and tensor.shape[1] > 1:
            tensor = tensor.amax(dim=1)
        else:
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

def has_spatial_detail(att: np.ndarray) -> bool:
    att = np.asarray(att, dtype=np.float32)
    if att.ndim < 2 or att.shape[-2] <= 1 or att.shape[-1] <= 1:
        return False
    values = att[np.isfinite(att)]
    if values.size == 0:
        return False
    return float(values.max() - values.min()) > 1e-6

def prepare_display_map(att: np.ndarray, shape, floor_percentile: float) -> np.ndarray:
    if not has_spatial_detail(att):
        return np.zeros(shape, dtype=np.float32)
    display_map = resize_map_to_shape(att, shape).astype(np.float32)
    values = display_map[np.isfinite(display_map)]
    if values.size == 0:
        return np.zeros(shape, dtype=np.float32)
    floor_percentile = float(np.clip(floor_percentile, 0.0, 99.0))
    floor = float(np.percentile(values, floor_percentile))
    display_map = np.maximum(display_map - floor, 0.0)
    return np.nan_to_num(display_map, nan=0.0, posinf=0.0, neginf=0.0)


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


def slice_base(image: torch.Tensor, map_index: int, num_maps: int) -> np.ndarray:
    """Return the pre-contrast image slice aligned with one slice-attention map."""
    if not torch.is_tensor(image) or image.ndim < 3:
        return center_pre(image)

    image_slices = int(image.shape[0])
    if image_slices <= 0:
        return center_pre(image)

    if image_slices == num_maps:
        image_index = map_index
    else:
        map_center = num_maps // 2
        image_center = image_slices // 2
        image_index = image_center + (map_index - map_center)
    image_index = int(np.clip(image_index, 0, image_slices - 1))
    return image[image_index, 0].numpy()


def save_attention_group(output_root: Path, stem: str, base: np.ndarray, maps: np.ndarray, prefix: str, sample, args):
    if maps.size == 0:
        return False

    indices = center_triplet_indices(maps.shape[0])
    entries = []
    for index in indices:
        offset = index - (maps.shape[0] // 2)
        att = maps[index]
        current_base = normalize_to_uint8(slice_base(sample["image"], index, maps.shape[0]))
        display_map = prepare_display_map(att, current_base.shape[:2], args.slice_display_floor_percentile)
        entries.append(
            {
                "offset": offset,
                "display_label": display_slice_label(offset),
                "name": safe_name(slice_file_suffix(offset)),
                "att": att,
                "base": current_base,
                "display_map": display_map,
            }
        )
    if not entries:
        return False

    display_maps = np.stack([entry["display_map"] for entry in entries], axis=0)
    vmin = float(args.slice_vmin)
    vmax = resolve_vmax(display_maps, args.slice_vmax, args.slice_vmax_percentile)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    panels = [add_title(gray_to_bgr(base), "center pre")]
    panels.extend(prediction_context_panels(sample, base))
    for entry in entries:
        overlay = attention_overlay_boosted(
            entry["base"],
            entry["display_map"],
            vmin=vmin,
            vmax=vmax,
            alpha=float(np.clip(args.slice_overlay_alpha, 0.0, 1.0)),
            gamma=args.slice_colormap_gamma,
            boost_start=args.slice_boost_start,
        )
        panels.append(add_title(overlay, f"{prefix} {entry['display_label']} mean={float(entry['att'].mean()):.3f}"))
        cv2.imwrite(str(output_root / f"{stem}_{prefix}_{entry['name']}_attention.png"), overlay)
        if args.save_raw_attention:
            raw_root = output_root / "raw_maps"
            raw_root.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(raw_root / f"{stem}_{prefix}_{entry['name']}_attention_raw.png"),
                normalize_to_uint8_fixed(entry["display_map"], vmin=vmin, vmax=vmax),
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
    parser.add_argument(
        "--slice_display_floor_percentile",
        type=float,
        default=60.0,
        help="Percentile subtracted from each slice-attention map before coloring. Higher values suppress broad background tint. Default: 60.",
    )
    parser.add_argument(
        "--slice_reduce",
        choices=("mean", "max", "phase"),
        default="mean",
        help="How to reduce slice attention across DCE phases. Use 'phase' with --slice_phase_index to inspect one phase instead of averaging all phases.",
    )
    parser.add_argument(
        "--slice_phase_index",
        type=int,
        default=None,
        help="DCE phase index used when --slice_reduce phase. Default: middle available phase.",
    )
    parser.add_argument(
        "--save_raw_attention",
        action="store_true",
        help="Also save grayscale normalized raw attention maps under raw_maps/. Default: off.",
    )
    args = parser.parse_args()

    saved = 0
    for sample in iter_outputs(args, "Slice attention visualization"):
        output_root = Path(sample["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        stem = Path(sample["name"]).stem
        base = normalize_to_uint8(center_pre(sample["image"]))
        maps = sample["output"].get("slice_attention_maps", [])
        image_slice_attn = reduce_slice_attention(maps[0] if maps else None, args.slice_reduce, args.slice_phase_index)
        kinetic_slice_attn = reduce_slice_attention(maps[1] if len(maps) > 1 else None, args.slice_reduce, args.slice_phase_index)
        wrote_image = save_attention_group(output_root, stem, base, image_slice_attn, "image_slice", sample, args)
        wrote_kinetic = save_attention_group(output_root, stem, base, kinetic_slice_attn, "kinetic_slice", sample, args)
        if wrote_image or wrote_kinetic:
            saved += 1
    print(f"Saved slice attention visualizations for {saved} samples")


if __name__ == "__main__":
    main()
