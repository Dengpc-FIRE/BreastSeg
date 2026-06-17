from __future__ import annotations

import csv
import math
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


METRIC_KEYS = ["dice", "iou", "hd95", "sensitivity", "precision", "accuracy"]


_NDI = None
_NDI_CHECKED = False


def _get_ndi():
    global _NDI, _NDI_CHECKED
    if os.environ.get("BREASTDM17_USE_SCIPY_HD95", "0") != "1":
        _NDI_CHECKED = True
        _NDI = None
        return None
    if _NDI_CHECKED:
        return _NDI
    _NDI_CHECKED = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy import ndimage as ndi
        _NDI = ndi
    except Exception:
        _NDI = None
    return _NDI


def _binary_erosion_numpy(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for axis in range(mask.ndim):
        lower = [slice(1, dim + 1) for dim in mask.shape]
        upper = [slice(1, dim + 1) for dim in mask.shape]
        lower[axis] = slice(0, mask.shape[axis])
        upper[axis] = slice(2, mask.shape[axis] + 2)
        eroded &= padded[tuple(lower)]
        eroded &= padded[tuple(upper)]
    return eroded & mask


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    ndi = _get_ndi()
    if ndi is None:
        return mask ^ _binary_erosion_numpy(mask)
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded


def _diagonal(shape: Iterable[int], spacing: Iterable[float] | None = None) -> float:
    spacing_tuple = tuple(float(s) for s in (spacing or [1.0] * len(tuple(shape))))
    return math.sqrt(sum(((dim - 1) * sp) ** 2 for dim, sp in zip(shape, spacing_tuple)))


def _hd95(pred: np.ndarray, target: np.ndarray, spacing: Iterable[float] | None = None) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if pred.any() != target.any():
        return _diagonal(pred.shape, spacing)

    pred_surface = _surface(pred)
    target_surface = _surface(target)
    if not pred_surface.any() or not target_surface.any():
        return 0.0 if np.array_equal(pred, target) else _diagonal(pred.shape, spacing)

    ndi = _get_ndi()
    if ndi is not None:
        spacing_tuple = tuple(float(s) for s in (spacing or [1.0] * pred.ndim))
        pred_dt = ndi.distance_transform_edt(~pred_surface, sampling=spacing_tuple)
        target_dt = ndi.distance_transform_edt(~target_surface, sampling=spacing_tuple)
        distances = np.concatenate([pred_dt[target_surface], target_dt[pred_surface]])
        return float(np.percentile(distances, 95)) if distances.size else 0.0

    pred_pts = np.argwhere(pred_surface)
    target_pts = np.argwhere(target_surface)
    if pred_pts.size == 0 or target_pts.size == 0:
        return _diagonal(pred.shape, spacing)
    if pred_pts.shape[0] * target_pts.shape[0] > 5_000_000:
        pred_pts = pred_pts[:: max(1, pred_pts.shape[0] // 3000)]
        target_pts = target_pts[:: max(1, target_pts.shape[0] // 3000)]
    distances = []
    for points_a, points_b in ((pred_pts, target_pts), (target_pts, pred_pts)):
        chunk_min = []
        for i in range(0, points_a.shape[0], 512):
            diff = points_a[i : i + 512, None, :] - points_b[None, :, :]
            chunk_min.append(np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1))
        distances.append(np.concatenate(chunk_min))
    return float(np.percentile(np.concatenate(distances), 95))


def compute_case_metrics(pred: np.ndarray, target: np.ndarray, spacing: Iterable[float] | None = None) -> Dict[str, float]:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {target.shape}")

    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    tn = float(np.logical_and(~pred, ~target).sum())
    pred_empty = not pred.any()
    target_empty = not target.any()

    if pred_empty and target_empty:
        dice = iou = sensitivity = precision = 1.0
    else:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "hd95": float(_hd95(pred, target, spacing)),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "accuracy": float(accuracy),
    }


def summarize_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {f"mean_{key}": float("nan") for key in METRIC_KEYS}
    return {f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}


def write_metrics_csv(path: str | Path, rows: List[Dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id"] + METRIC_KEYS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(path: str | Path, summary: Dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in [
            "mean_dice",
            "mean_iou",
            "mean_hd95",
            "mean_sensitivity",
            "mean_precision",
            "mean_accuracy",
        ]:
            handle.write(f"{key}: {summary[key]:.6f}\n")
