from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from visual.common import (
    gray_to_bgr,
    heatmap_overlay_fixed,
    normalize_to_uint8,
    normalize_to_uint8_fixed,
)

PLASMA_COLORMAP = getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET)


def tensor_to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def slice_window_indices(count: int) -> Sequence[int]:
    if count <= 0:
        return []
    center = count // 2
    return [max(0, center - 1), center, min(count - 1, center + 1)]


def _valid_indices(indices, count: int):
    result = []
    for index in indices or []:
        index = int(index)
        if 0 <= index < count:
            result.append(index)
    return result


def phase_display_indices(config: Dict, count: int) -> Sequence[int]:
    model_cfg = config.get("model", {})
    phase_cfg = model_cfg.get("phase_indices", {})
    pre = int(phase_cfg.get("pre", 0)) if count else 0
    pre = min(max(pre, 0), max(count - 1, 0))
    post = _valid_indices(phase_cfg.get("post", []), count)
    sub = _valid_indices(phase_cfg.get("subtraction", []), count)
    if post:
        early = post[0]
        middle = post[len(post) // 2]
        late = post[-1]
    else:
        early = min(1, max(count - 1, 0))
        middle = min(2, max(count - 1, 0))
        late = min(3, max(count - 1, 0))
    subtraction = sub[-1] if sub else min(4, max(count - 1, 0))
    return [pre, early, middle, late, subtraction]


def kinetic_display_indices(config: Dict, count: int) -> Sequence[int]:
    model_cfg = config.get("model", {})
    phase_cfg = model_cfg.get("phase_indices", {})
    pre = int(phase_cfg.get("pre", 0)) if count else 0
    pre = min(max(pre, 0), max(count - 1, 0))
    post = _valid_indices(phase_cfg.get("post", []), count)
    sub = _valid_indices(phase_cfg.get("subtraction", []), count)
    post1 = post[0] if len(post) > 0 else min(1, max(count - 1, 0))
    post2 = post[1] if len(post) > 1 else post1
    post3 = post[2] if len(post) > 2 else post2
    subtraction = sub[min(2, len(sub) - 1)] if sub else min(4, max(count - 1, 0))
    return [pre, post1, post2, post3, subtraction]


def phase_panel(image: np.ndarray, slice_index: int, phase_index: int) -> np.ndarray:
    slice_index = int(np.clip(slice_index, 0, image.shape[0] - 1))
    phase_index = int(np.clip(phase_index, 0, image.shape[1] - 1))
    return gray_to_bgr(normalize_to_uint8(image[slice_index, phase_index]))


def overlay_attention(base_gray: np.ndarray, value_map: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    return heatmap_overlay_fixed(normalize_to_uint8(base_gray), value_map, vmin=vmin, vmax=vmax)


def prior_heatmap(value_map: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(normalize_to_uint8(value_map), PLASMA_COLORMAP)


def fit_panel(panel: np.ndarray, size: Tuple[int, int], fill: int = 0) -> np.ndarray:
    panel = panel if panel.ndim == 3 else gray_to_bgr(panel)
    target_h, target_w = int(size[0]), int(size[1])
    h, w = panel.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(panel, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    y = (target_h - new_h) // 2
    x = (target_w - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def put_centered_text(canvas: np.ndarray, text: str, x0: int, x1: int, y: int, font_scale: float, thickness: int = 1) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = str(text).split("\n")
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_h = max((size[1] for size in sizes), default=10) + 5
    start_y = y - (len(lines) - 1) * line_h // 2
    for offset, (line, (tw, th)) in enumerate(zip(lines, sizes)):
        x = int(x0 + max(0, (x1 - x0 - tw) // 2))
        cv2.putText(canvas, line, (x, start_y + offset * line_h + th // 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def paste_vertical_text(canvas: np.ndarray, text: str, x: int, y0: int, y1: int) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    label = np.full((th + 12, tw + 12, 3), 255, dtype=np.uint8)
    cv2.putText(label, text, (6, th + 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = label.shape[:2]
    yy = int(y0 + max(0, (y1 - y0 - h) // 2))
    y_end = min(canvas.shape[0], yy + h)
    x_end = min(canvas.shape[1], x + w)
    if yy < y_end and x < x_end:
        canvas[yy:y_end, x:x_end] = label[: y_end - yy, : x_end - x]


def draw_colorbar(canvas: np.ndarray, x: int, y: int, height: int, width: int, colormap: int, label: str, tick_labels=None) -> None:
    gradient = np.linspace(255, 0, max(1, int(height)), dtype=np.uint8)[:, None]
    bar = cv2.applyColorMap(np.repeat(gradient, int(width), axis=1), colormap)
    canvas[y : y + height, x : x + width] = bar
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (90, 90, 90), 1)
    if tick_labels is None:
        tick_labels = [(1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")]
    for value, text in tick_labels:
        value = float(np.clip(value, 0.0, 1.0))
        yy = int(round(y + (1.0 - value) * height))
        cv2.line(canvas, (x + width, yy), (x + width + 5, yy), (0, 0, 0), 1)
        cv2.putText(canvas, text, (x + width + 10, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    paste_vertical_text(canvas, label, x + width + 54, y, y + height)


def compose_case_matrix(
    title: str,
    rows: Sequence[Sequence[np.ndarray]],
    column_labels: Sequence[str],
    output_path: Path,
    panel_size: Tuple[int, int] = (150, 150),
    gap: int = 6,
    colorbar: Optional[Dict] = None,
) -> np.ndarray:
    if not rows:
        raise ValueError("No rows to compose")
    n_rows = len(rows)
    n_cols = len(column_labels)
    if any(len(row) != n_cols for row in rows):
        raise ValueError("Row panel count does not match column labels")

    panel_h, panel_w = int(panel_size[0]), int(panel_size[1])
    left_w = 82
    top_h = 72
    header_h = 30
    bottom_pad = 24
    matrix_w = n_cols * panel_w + (n_cols - 1) * gap
    matrix_h = n_rows * panel_h + (n_rows - 1) * gap
    right_w = 126 if colorbar else 18
    width = left_w + matrix_w + right_w
    height = top_h + header_h + matrix_h + bottom_pad
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    put_centered_text(canvas, title, left_w, left_w + matrix_w, 26, 1.05, thickness=2)
    for col, label in enumerate(column_labels):
        x0 = left_w + col * (panel_w + gap)
        put_centered_text(canvas, label, x0, x0 + panel_w, top_h + 4, 0.55, thickness=2)

    y0 = top_h + header_h
    for row_idx, panels in enumerate(rows):
        y = y0 + row_idx * (panel_h + gap)
        put_centered_text(canvas, f"Case {row_idx + 1}", 0, left_w - 8, y + panel_h // 2 - 8, 0.5, thickness=2)
        for col_idx, panel in enumerate(panels):
            x = left_w + col_idx * (panel_w + gap)
            canvas[y : y + panel_h, x : x + panel_w] = fit_panel(panel, (panel_h, panel_w))

    if colorbar:
        cb_x = left_w + matrix_w + 30
        cb_h = max(110, int(matrix_h * 0.8))
        cb_y = y0 + max(0, (matrix_h - cb_h) // 2)
        draw_colorbar(
            canvas,
            cb_x,
            cb_y,
            cb_h,
            int(colorbar.get("width", 28)),
            int(colorbar.get("colormap", cv2.COLORMAP_JET)),
            str(colorbar.get("label", "")),
            colorbar.get("tick_labels"),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return canvas


def save_fixed_map(path: Path, image: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), normalize_to_uint8_fixed(image, vmin=vmin, vmax=vmax))
