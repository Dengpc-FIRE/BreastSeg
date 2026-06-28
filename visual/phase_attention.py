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
    write_csv,
)
from visual.paper_style import (
    compose_case_matrix,
    overlay_attention,
    phase_display_indices,
    save_fixed_map,
    tensor_to_numpy,
)

COLUMN_LABELS = ["Pre", "Early", "Middle", "Late", "Subtraction"]


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


def build_phase_row(sample, attn: np.ndarray, stem: str, output_root: Path, vmin: float, vmax: float):
    if attn.size == 0:
        return None, []
    image = tensor_to_numpy(sample["image"])
    center_slice = image.shape[0] // 2
    count = min(attn.shape[0], image.shape[1])
    indices = phase_display_indices(sample["config"], count)
    rows = []
    panels = []
    gt = tensor_to_numpy(sample["mask"])[0] >= 0.5

    for label, phase_index in zip(COLUMN_LABELS, indices):
        base = image[center_slice, phase_index]
        display_map = resize_map_to_shape(attn[phase_index], base.shape[:2])
        panels.append(overlay_attention(base, display_map, vmin=vmin, vmax=vmax))
        tumor_mean = float(display_map[gt].mean()) if gt.any() else float("nan")
        background_mean = float(display_map[~gt].mean()) if (~gt).any() else float("nan")
        rows.append(
            {
                "name": sample["name"],
                "phase": label,
                "phase_index": int(phase_index),
                "mean": float(display_map.mean()),
                "tumor_mean": tumor_mean,
                "background_mean": background_mean,
                "vmin": float(vmin),
                "vmax": float(vmax),
            }
        )
        save_fixed_map(output_root / f"{stem}_{label.lower()}_attention.png", display_map, vmin=vmin, vmax=vmax)
    return panels, rows


def main():
    parser = argparse.ArgumentParser(description="Visualize phase attention maps in a paper-style case matrix.")
    add_common_args(parser, "phase_attention")
    parser.add_argument("--phase_vmin", type=float, default=0.0, help="Lower bound for the shared phase-attention colormap.")
    parser.add_argument("--phase_vmax", type=float, default=1.0, help="Upper bound for the shared phase-attention colormap.")
    parser.add_argument("--panel_size", type=int, default=150, help="Square panel size in pixels for the summary figure.")
    args = parser.parse_args()

    rows = []
    csv_rows = []
    last_output_root = None
    for sample in iter_outputs(args, "Phase attention visualization"):
        output_root = Path(sample["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        last_output_root = output_root
        stem = Path(sample["name"]).stem
        attn = extract_phase_attention(sample["output"])
        if attn.size == 0:
            continue
        vmax = float(args.phase_vmax)
        vmin = float(args.phase_vmin)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        panels, sample_rows = build_phase_row(sample, attn, stem, output_root, vmin=vmin, vmax=vmax)
        if panels is not None:
            rows.append(panels)
            csv_rows.extend(sample_rows)

    if last_output_root is not None and rows:
        compose_case_matrix(
            "Phase Attention Visualization",
            rows,
            COLUMN_LABELS,
            last_output_root / "phase_attention_visualization.png",
            panel_size=(int(args.panel_size), int(args.panel_size)),
            colorbar={
                "colormap": cv2.COLORMAP_JET,
                "label": "Phase Attention Weight",
                "tick_labels": [(1.0, "1.0"), (0.0, "0.0")],
            },
        )
    if csv_rows and last_output_root is not None:
        write_csv(
            last_output_root / "phase_attention_weights.csv",
            csv_rows,
            ["name", "phase", "phase_index", "mean", "tumor_mean", "background_mean", "vmin", "vmax"],
        )
    print(f"Saved phase attention summary for {len(rows)} samples")


if __name__ == "__main__":
    main()
