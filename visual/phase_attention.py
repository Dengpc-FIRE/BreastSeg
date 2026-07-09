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
    phase_labels,
    prediction_context_panels,
    resize_map_to_shape,
    write_csv,
)


def safe_phase_name(label: str) -> str:
    label = str(label).strip().lower().replace(" ", "_").replace("-", "")
    label = re.sub(r"[^0-9a-zA-Z_]+", "_", label)
    return label.strip("_") or "phase"


def extract_phase_attention(output) -> np.ndarray:
    attn = output.get("phase_attention")
    if attn is None:
        maps = output.get("phase_attention_maps") or output.get("attention_maps") or []
        attn = maps[0] if maps else None
    if attn is None or not torch.is_tensor(attn):
        return np.empty((0, 0, 0), dtype=np.float32)
    tensor = attn.detach().float()
    if tensor.ndim == 5:
        # Usually slice_output_dict removes B. Keep this robust for direct debug calls.
        tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor.mean(dim=0)
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            # [T,1,H,W] -> [T,H,W].
            tensor = tensor.squeeze(1)
        elif tensor.shape[0] == 1:
            # [1,T,H,W] -> [T,H,W].
            tensor = tensor.squeeze(0)
        else:
            # [T,C,H,W] -> [T,H,W].
            tensor = tensor.mean(dim=1)
    elif tensor.ndim != 3:
        return np.empty((0, 0, 0), dtype=np.float32)
    return tensor.numpy()


def draw_bar(values: np.ndarray, labels, width: int = 760, height: int = 260) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_b, margin_t = 54, 50, 24
    plot_w = width - margin_l - 18
    plot_h = height - margin_t - margin_b
    max_v = max(float(values.max()), 1e-6)
    bar_w = max(4, int(plot_w / max(len(values), 1) * 0.65))
    for i, value in enumerate(values):
        x = margin_l + int((i + 0.18) * plot_w / len(values))
        h = int(plot_h * float(value) / max_v)
        y = margin_t + plot_h - h
        cv2.rectangle(canvas, (x, y), (x + bar_w, margin_t + plot_h), (42, 97, 219), -1)
        cv2.putText(canvas, f"{value:.3f}", (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 30, 30), 1)
        cv2.putText(canvas, labels[i], (x - 5, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1)
    cv2.line(canvas, (margin_l, margin_t), (margin_l, margin_t + plot_h), (0, 0, 0), 1)
    cv2.line(canvas, (margin_l, margin_t + plot_h), (width - 8, margin_t + plot_h), (0, 0, 0), 1)
    return canvas


def boost_attention_colormap(norm: np.ndarray, gamma: float, start: float) -> np.ndarray:
    """Push mid/high attention values toward the red end without changing vmax."""
    norm = np.clip(norm.astype(np.float32), 0.0, 1.0)
    start = float(np.clip(start, 0.0, 0.95))
    gamma = max(float(gamma), 1e-6)
    boosted = norm.copy()
    active = norm > start
    if np.any(active):
        local = (norm[active] - start) / max(1.0 - start, 1e-6)
        boosted[active] = start + (1.0 - start) * np.power(local, gamma)
    return np.clip(boosted, 0.0, 1.0)


def attention_overlay_boosted(
    base: np.ndarray,
    display_map: np.ndarray,
    vmin: float,
    vmax: float,
    alpha: float,
    gamma: float,
    boost_start: float,
) -> np.ndarray:
    norm = normalize_to_uint8_fixed(display_map, vmin=vmin, vmax=vmax).astype(np.float32) / 255.0
    norm = boost_attention_colormap(norm, gamma=gamma, start=boost_start)
    heatmap = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(gray_to_bgr(base), 1.0 - alpha, heatmap, alpha, 0)


def resolve_vmax(attn: np.ndarray, explicit_vmax, percentile):
    if explicit_vmax is not None:
        return float(explicit_vmax)
    values = np.asarray(attn, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    if percentile is None:
        return float(values.max())
    percentile = float(np.clip(percentile, 50.0, 100.0))
    return float(np.percentile(values, percentile))


def main():
    parser = argparse.ArgumentParser(description="Visualize all pixel-wise phase attention maps with a shared colormap scale.")
    add_common_args(parser, "phase_attention")
    parser.add_argument(
        "--phase_vmin",
        type=float,
        default=0.0,
        help="Lower bound for the shared phase-attention colormap. Default: 0.",
    )
    parser.add_argument(
        "--phase_vmax",
        type=float,
        default=None,
        help="Upper bound for the shared phase-attention colormap. Default: per-sample max attention.",
    )
    parser.add_argument(
        "--phase_vmax_percentile",
        type=float,
        default=None,
        help="Optional percentile used as phase_vmax when --phase_vmax is not set. Lower values make hotspots redder.",
    )
    parser.add_argument(
        "--phase_overlay_alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity. Default matches the original phase_attention_grid style: 0.45.",
    )
    parser.add_argument(
        "--phase_colormap_gamma",
        type=float,
        default=0.55,
        help="Nonlinear boost for mid/high attention colors. Lower values make yellow regions redder. Default: 0.55.",
    )
    parser.add_argument(
        "--phase_boost_start",
        type=float,
        default=0.25,
        help="Normalized attention value where red-boost starts. Lower values affect more area. Default: 0.25.",
    )
    args = parser.parse_args()

    saved = 0
    rows = []
    last_output_root = None
    for sample in iter_outputs(args, "Phase attention visualization"):
        output_root = Path(sample["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        last_output_root = output_root
        stem = Path(sample["name"]).stem
        base = normalize_to_uint8(center_pre(sample["image"]))
        attn = extract_phase_attention(sample["output"])
        if attn.size == 0:
            continue

        labels = phase_labels(sample["config"], attn.shape[0])
        mean_weights = attn.reshape(attn.shape[0], -1).mean(axis=1)
        vmin = float(args.phase_vmin)
        vmax = resolve_vmax(attn, args.phase_vmax, args.phase_vmax_percentile)
        if vmax <= vmin:
            vmax = vmin + 1e-6

        gt = sample["mask"][0].numpy() >= 0.5
        panels = [add_title(gray_to_bgr(base), "center pre")]
        panels.extend(prediction_context_panels(sample, base))
        for index, phase_map in enumerate(attn):
            display_map = resize_map_to_shape(phase_map, base.shape[:2])
            overlay = attention_overlay_boosted(
                base,
                display_map,
                vmin=vmin,
                vmax=vmax,
                alpha=float(np.clip(args.phase_overlay_alpha, 0.0, 1.0)),
                gamma=args.phase_colormap_gamma,
                boost_start=args.phase_boost_start,
            )
            tumor_mean = float(display_map[gt].mean()) if gt.any() else float("nan")
            background_mean = float(display_map[~gt].mean()) if (~gt).any() else float("nan")
            rows.append(
                {
                    "name": sample["name"],
                    "phase": labels[index],
                    "mean": float(display_map.mean()),
                    "tumor_mean": tumor_mean,
                    "background_mean": background_mean,
                    "vmin": vmin,
                    "vmax": vmax,
                }
            )
            panels.append(add_title(overlay, f"{labels[index]} mean={mean_weights[index]:.3f}"))
            phase_name = safe_phase_name(labels[index])
            cv2.imwrite(str(output_root / f"{stem}_{phase_name}_attention.png"), overlay)
            cv2.imwrite(
                str(output_root / f"{stem}_{phase_name}_attention_raw.png"),
                normalize_to_uint8_fixed(display_map, vmin=vmin, vmax=vmax),
            )
        panels.append(add_title(draw_bar(mean_weights, labels), "average phase weights"))
        cv2.imwrite(str(output_root / f"{stem}_phase_attention_grid.png"), make_grid(panels, cols=4))
        saved += 1

    if rows and last_output_root is not None:
        write_csv(
            last_output_root / "phase_attention_weights.csv",
            rows,
            ["name", "phase", "mean", "tumor_mean", "background_mean", "vmin", "vmax"],
        )
    print(f"Saved phase attention visualizations for {saved} samples")


if __name__ == "__main__":
    main()
