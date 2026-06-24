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
    heatmap_overlay_fixed,
    iter_outputs,
    make_grid,
    normalize_to_uint8,
    normalize_to_uint8_fixed,
    phase_labels,
    prediction_context_panels,
    resize_map_to_shape,
    write_csv,
)


def extract_phase_attention(output) -> np.ndarray:
    attn = output.get("phase_attention")
    if attn is None:
        maps = output.get("phase_attention_maps") or output.get("attention_maps") or []
        attn = maps[0] if maps else None
    if attn is None or not torch.is_tensor(attn):
        return np.empty((0, 0, 0), dtype=np.float32)
    tensor = attn.detach().float()
    if tensor.ndim == 4:
        # [T,1,H,W] -> [T,H,W].
        tensor = tensor.squeeze(1)
    elif tensor.ndim == 3:
        pass
    else:
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
        vmax = float(args.phase_vmax) if args.phase_vmax is not None else float(np.nanmax(attn))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        gt = (sample["mask"][0].numpy() >= 0.5)
        panels = [add_title(gray_to_bgr(base), "center pre")]
        panels.extend(prediction_context_panels(sample, base))
        for index, phase_map in enumerate(attn):
            display_map = resize_map_to_shape(phase_map, base.shape[:2])
            overlay = heatmap_overlay_fixed(base, display_map, vmin=vmin, vmax=vmax)
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
            cv2.imwrite(
                str(output_root / f"{stem}_{labels[index]}_attention.png"),
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
