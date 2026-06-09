from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from reproduce_pdfunet.utils.split import infer_patient_id


METRIC_KEYS = ["dice", "iou", "recall", "precision", "accuracy", "hd95"]


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray, hd95_empty_value: float = 128.0) -> Dict[str, float]:
    pred = np.squeeze((pred > 0).astype(np.uint8))
    gt = np.squeeze((gt > 0).astype(np.uint8))
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return {"dice": 1.0, "iou": 1.0, "recall": 1.0, "precision": 1.0, "accuracy": 1.0, "hd95": 0.0}
    if pred_sum == 0 or gt_sum == 0:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "accuracy": float((pred == gt).mean()),
            "hd95": float(hd95_empty_value),
        }
    return {**_confusion_metrics(pred, gt), "hd95": hd95(pred, gt, hd95_empty_value)}


def hd95(pred: np.ndarray, gt: np.ndarray, hd95_empty_value: float = 128.0) -> float:
    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    if not pred_surface.any() and not gt_surface.any():
        return 0.0
    if not pred_surface.any() or not gt_surface.any():
        return float(hd95_empty_value)
    pred_to_gt = distance_transform_edt(~gt_surface)[pred_surface]
    gt_to_pred = distance_transform_edt(~pred_surface)[gt_surface]
    distances = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float32)
    return float(np.percentile(distances, 95))


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


def global_pixel_metrics(preds: Iterable[np.ndarray], gts: Iterable[np.ndarray]) -> Dict[str, float]:
    pred = np.concatenate([(p > 0).astype(np.uint8).reshape(-1) for p in preds])
    gt = np.concatenate([(g > 0).astype(np.uint8).reshape(-1) for g in gts])
    return {**_confusion_metrics(pred, gt), "hd95": float("nan")}


def mean_std(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([fold[key] for fold in fold_metrics])),
            "std": float(np.std([fold[key] for fold in fold_metrics], ddof=0)),
        }
        for key in METRIC_KEYS
    }


def _confusion_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() == 0 and gt.sum() == 0:
        return {"dice": 1.0, "iou": 1.0, "recall": 1.0, "precision": 1.0, "accuracy": 1.0}
    if pred.sum() == 0 or gt.sum() == 0:
        return {"dice": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0, "accuracy": float((pred == gt).mean())}
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
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.squeeze(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"HD95 expects a 2D mask, got shape {mask.shape}")
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask ^ eroded

