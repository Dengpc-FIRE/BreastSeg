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
    iter_outputs,
    normalize_to_uint8_fixed,
    phase_labels,
    resize_map_to_shape,
    write_csv,
)
from visual.paper_style import (
    compose_case_matrix,
    overlay_attention,
    phase_display_indices,
    tensor_to_numpy,
)

COLUMN_LABELS = ["Pre", "Early", "Middle", "Late", "Subtraction"]


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
        # [B,T,1,H,W] should normally be sliced before this script sees it,
        # but keep the extractor robust for direct debug calls.
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


def save_phase_overlays(sample, attn: np.ndarray, stem: str, output_root: Path, vmin: float, vmax: float):
    """Save every phase as a colored attention overlay and return overlay panels."""
    image = tensor_to_numpy(sample["image"])
    center_slice = image.shape[0] // 2
    count = min(attn.shape[0], image.shape[1])
    labels = phase_labels(sample["config"], count)
    overlays = []
    maps = []

    for phase_index in range(count):
        label = safe_phase_name(labels[phase_index])
        base = image[center_slice, phase_index]
        display_map = resize_map_to_shape(attn[phase_index], base.shape[:2])
        overlay = overlay_attention(base, display_map, vmin=vmin, vmax=vmax)
        overlays.append(overlay)
        maps.append(display_map)
        cv2.imwrite(str(output_root / f"{stem}_{label}_attention.png"), overlay)
        cv2.imwrite(
            str(output_root / f"{stem}_{label}_attention_raw.png"),
            normalize_to_uint8_fixed(display_map, vmin=vmin, vmax=vmax),
        )
    return overlays, maps, labels


def build_phase_row(sample, attn: np.ndarray, stem: str, output_root: Path, vmin: float, vmax: float):
    if attn.size == 0:
        return None, []
    image = tensor_to_numpy(sample["image"])
    count = min(attn.shape[0], image.shape[1])
    if count <= 0:
        return None, []

    overlays, maps, labels = save_phase_overlays(sample, attn, stem, output_root, vmin=vmin, vmax=vmax)
    indices = phase_display_indices(sample["config"], count)
    gt = tensor_to_numpy(sample["mask"])[0] >= 0.5
    panels = []
    rows = []

    for display_label, phase_index in zip(COLUMN_LABELS, indices):
        phase_index = int(np.clip(phase_index, 0, count - 1))
        display_map = maps[phase_index]
        panels.append(overlays[phase_index])
        tumor_mean = float(display_map[gt].mean()) if gt.any() else float("nan")
        background_mean = float(display_map[~gt].mean()) if (~gt).any() else float("nan")
        rows.append(
            {
                "name": sample["name"],
                "phase": display_label,
                "source_phase": labels[phase_index],
                "phase_index": int(phase_index),
                "mean": float(display_map.mean()),
                "tumor_mean": tumor_mean,
                "background_mean": background_mean,
                "vmin": float(vmin),
                "vmax": float(vmax),
            }
        )
    return panels, rows


def resolve_vmax(attn: np.ndarray, vmin: float, explicit_vmax, percentile: float):
    if explicit_vmax is not None:
        vmax = float(explicit_vmax)
    else:
        values = np.nan_to_num(np.asarray(attn, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        positive = values[values > vmin]
        if positive.size:
            percentile = float(np.clip(percentile, 50.0, 100.0))
            vmax = float(np.percentile(positive, percentile))
        else:
            vmax = float(values.max()) if values.size else vmin + 1e-6
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    return vmax


def main():
    parser = argparse.ArgumentParser(description="Visualize phase attention maps in a paper-style case matrix.")
    add_common_args(parser, "phase_attention")
    parser.add_argument("--phase_vmin", type=float, default=0.0, help="Lower bound for the shared phase-attention colormap.")
    parser.add_argument(
        "--phase_vmax",
        type=float,
        default=None,
        help="Upper bound for the phase-attention colormap. Default: per-sample percentile attention for visible overlays.",
    )
    parser.add_argument(
        "--phase_vmax_percentile",
        type=float,
        default=99.0,
        help="Percentile used as phase_vmax when --phase_vmax is not set. Default: 99.",
    )
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
        vmin = float(args.phase_vmin)
        vmax = resolve_vmax(attn, vmin, args.phase_vmax, args.phase_vmax_percentile)
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
                "tick_labels": [(1.0, "High"), (0.0, "Low")],
            },
        )
    if csv_rows and last_output_root is not None:
        write_csv(
            last_output_root / "phase_attention_weights.csv",
            csv_rows,
            ["name", "phase", "source_phase", "phase_index", "mean", "tumor_mean", "background_mean", "vmin", "vmax"],
        )
    print(f"Saved phase attention summary for {len(rows)} samples")


if __name__ == "__main__":
    main()
