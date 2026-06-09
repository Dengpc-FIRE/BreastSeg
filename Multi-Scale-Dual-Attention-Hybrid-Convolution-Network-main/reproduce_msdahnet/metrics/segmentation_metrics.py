from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff

from reproduce_msdahnet.utils.split import infer_patient_id


METRIC_KEYS = ["dice", "iou", "recall", "precision", "accuracy", "hd"]


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray, hd_empty_value: float = 256.0) -> Dict[str, float]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return {"dice": 1.0, "iou": 1.0, "recall": 1.0, "precision": 1.0, "accuracy": 1.0, "hd": 0.0}
    if pred_sum == 0 or gt_sum == 0:
        accuracy = float((pred == gt).mean())
        return {"dice": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0, "accuracy": accuracy, "hd": hd_empty_value}

    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())
    tn = float(((1 - pred) * (1 - gt)).sum())
    eps = 1e-8
    return {
        "dice": (2.0 * tp) / (2.0 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "recall": tp / (tp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "hd": hausdorff_distance(pred, gt, hd_empty_value=hd_empty_value),
    }


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray, hd_empty_value: float = 256.0) -> float:
    pred_pts = _boundary_points(pred)
    gt_pts = _boundary_points(gt)
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float(hd_empty_value)
    return float(max(directed_hausdorff(pred_pts, gt_pts)[0], directed_hausdorff(gt_pts, pred_pts)[0]))


def global_pixel_metrics(preds: Iterable[np.ndarray], gts: Iterable[np.ndarray], hd_empty_value: float = 256.0) -> Dict[str, float]:
    pred = np.concatenate([(p > 0).astype(np.uint8).reshape(-1) for p in preds])
    gt = np.concatenate([(g > 0).astype(np.uint8).reshape(-1) for g in gts])
    metrics = compute_sample_metrics(pred, gt, hd_empty_value=hd_empty_value)
    metrics["hd"] = float("nan")
    return metrics


def summarize_slice_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}


def summarize_patient_metrics(rows: List[Dict], image_paths: List[str]) -> Dict[str, float]:
    patient_rows = defaultdict(list)
    for row, image_path in zip(rows, image_paths):
        patient_rows[infer_patient_id(image_path)].append(row)
    return {
        key: float(np.mean([np.mean([row[key] for row in values]) for values in patient_rows.values()]))
        for key in METRIC_KEYS
    }


def mean_std(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([fold[key] for fold in fold_metrics])),
            "std": float(np.std([fold[key] for fold in fold_metrics], ddof=0)),
        }
        for key in METRIC_KEYS
    }


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(bool)
    if not mask.any():
        return np.empty((0, 2), dtype=np.float32)
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    boundary = mask ^ eroded
    return np.argwhere(boundary).astype(np.float32)
