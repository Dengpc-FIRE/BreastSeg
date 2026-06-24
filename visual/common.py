"""Shared visualization utilities for BreastSeg.

The scripts in this directory are intentionally inference-only. They load a
trained checkpoint, run the tumor model with ``return_dict=True``, and save
human-readable panels for masks, attention, kinetic maps, uncertainty and
boundary outputs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.whole_breast_constraint import build_whole_breast_constraint
from train.train_config import (
    build_model_from_config,
    checkpoint_name_from_config,
    load_config,
    resolve_config_path,
)


PHASE_LABELS_17 = [
    "pre",
    "post1",
    "post2",
    "post3",
    "post4",
    "post5",
    "post6",
    "post7",
    "post8",
    "sub1",
    "sub2",
    "sub3",
    "sub4",
    "sub5",
    "sub6",
    "sub7",
    "sub8",
]


class KPTA25DVisualDataset(Dataset):
    """Read processed 2.5D samples stored as [K,T,H,W]."""

    def __init__(
        self,
        split_path: str,
        input_phase_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.split_path = Path(split_path)
        self.data_dir = self.split_path / "data"
        self.gt_dir = self.split_path / "GT"
        self.input_phase_indices = (
            [int(index) for index in input_phase_indices]
            if input_phase_indices is not None
            else None
        )
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")
        self.files = sorted(self.data_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy samples found in: {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        data_path = self.files[index]
        image = np.load(data_path).astype(np.float32)
        if image.ndim != 4:
            raise ValueError(
                f"Expected [K,T,H,W] input, got {image.shape} for {data_path.name}"
            )
        if self.input_phase_indices is not None:
            image = image[:, self.input_phase_indices]
        gt_path = self.gt_dir / f"{data_path.stem}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            gt = np.zeros(image.shape[-2:], dtype=np.uint8)
        height, width = image.shape[-2:]
        if gt.shape != (height, width):
            gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)
        gt = (gt > 127).astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(gt[None]), data_path.name


def add_common_args(parser: argparse.ArgumentParser, default_subdir: str) -> None:
    parser.add_argument("--config", default="configs/kpta_25d_net.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_subdir", default=default_subdir)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def resolve_checkpoint(config_path: str, config: Dict, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    train_cfg = config.get("train", {})
    output_path = Path(train_cfg.get("output_path", "./results_kpta_25d_net"))
    candidate = output_path / checkpoint_name_from_config(config_path)
    if candidate.is_file():
        return candidate
    legacy = output_path / "best_model.pth"
    if legacy.is_file():
        return legacy
    return candidate


def build_context(args: argparse.Namespace):
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    train_cfg = config.get("train", {})
    split_path = Path(args.split_path or train_cfg.get(f"{args.split}_path", ""))
    if not split_path:
        raise ValueError(f"Cannot resolve split path for split={args.split}")
    output_root = Path(
        args.output_dir
        or (
            Path(train_cfg.get("output_path", "./results_kpta_25d_net"))
            / "visual"
            / args.split
            / args.output_subdir
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_checkpoint(config_path, config, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    input_phase_indices = config.get("dataset", {}).get("input_phase_indices")
    dataset = KPTA25DVisualDataset(
        str(split_path),
        input_phase_indices=input_phase_indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
    )
    device = torch.device(args.device)
    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    whole_breast = build_whole_breast_constraint(
        config,
        device=device,
        output_path=Path(train_cfg.get("output_path", "./results_kpta_25d_net")),
    )
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split_path} ({len(dataset)} samples)")
    print(f"Output: {output_root}")
    print(f"Whole-breast constraint: {'enabled' if whole_breast else 'disabled'}")
    return config, dataset, loader, model, device, output_root, whole_breast


def iter_outputs(args: argparse.Namespace, desc: str):
    config, dataset, loader, model, device, output_root, whole_breast = build_context(args)
    max_samples = max(1, int(args.max_samples))
    count = 0
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for images, masks, names in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(images, return_dict=True)
                logits = output["seg_logits"] if isinstance(output, dict) else output
                probabilities = torch.sigmoid(logits)
            if whole_breast is not None:
                probabilities, breast_masks = whole_breast.constrain_probabilities(
                    probabilities,
                    names,
                    dataset,
                )
            else:
                breast_masks = torch.ones_like(probabilities)
            batch = images.shape[0]
            images_cpu = images.float().cpu()
            masks_cpu = masks.float().cpu()
            probabilities_cpu = probabilities.float().cpu()
            breast_cpu = breast_masks.float().cpu()
            output_cpu = to_cpu_dict(output)
            for index in range(batch):
                if count >= max_samples:
                    return
                yield {
                    "config": config,
                    "dataset": dataset,
                    "output_root": output_root,
                    "name": names[index],
                    "image": images_cpu[index],
                    "mask": masks_cpu[index],
                    "probability": probabilities_cpu[index],
                    "breast_mask": breast_cpu[index],
                    "output": slice_output_dict(output_cpu, index),
                    "threshold": float(args.threshold),
                }
                count += 1


def to_cpu_dict(output):
    if not isinstance(output, dict):
        return {"seg_logits": output.detach().float().cpu()}
    result = {}
    for key, value in output.items():
        if torch.is_tensor(value):
            result[key] = value.detach().float().cpu()
        elif isinstance(value, list):
            result[key] = [
                item.detach().float().cpu() if torch.is_tensor(item) else item
                for item in value
            ]
        elif isinstance(value, dict):
            result[key] = {
                k: v.detach().float().cpu() if torch.is_tensor(v) else v
                for k, v in value.items()
            }
        else:
            result[key] = value
    return result


def slice_output_dict(output: Dict, index: int) -> Dict:
    result = {}
    for key, value in output.items():
        if torch.is_tensor(value) and value.shape[:1] and value.shape[0] > index:
            result[key] = value[index]
        elif isinstance(value, list):
            items = []
            for item in value:
                if torch.is_tensor(item) and item.shape[:1] and item.shape[0] > index:
                    items.append(item[index])
                else:
                    items.append(item)
            result[key] = items
        elif isinstance(value, dict):
            nested = {}
            for k, v in value.items():
                if torch.is_tensor(v) and v.shape[:1] and v.shape[0] > index:
                    nested[k] = v[index]
                else:
                    nested[k] = v
            result[key] = nested
        else:
            result[key] = value
    return result


def normalize_to_uint8(image: np.ndarray, percentile: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32))
    low, high = np.percentile(image, percentile)
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def heatmap(image: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    return cv2.applyColorMap(normalize_to_uint8(image), colormap)


def normalize_to_uint8_fixed(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize with an explicit value range.

    This is useful for attention maps. If every phase is normalized separately,
    a low-weight subtraction map can still look red because its tiny internal
    variation is stretched to the full colormap. A common range preserves the
    relative strength across phases.
    """
    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    image = np.clip((image - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def heatmap_fixed(
    image: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    return cv2.applyColorMap(normalize_to_uint8_fixed(image, vmin, vmax), colormap)


def resize_map_to_shape(
    image: np.ndarray,
    spatial_shape: Tuple[int, int],
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Resize a scalar map to a target [H,W] shape for image overlays.

    Attention tensors are not always full-resolution. For example, CSAM slice
    attention is a global [K,1,1] weight, while simple slice attention is
    [K,H,W]. This helper converts both into displayable [H,W] maps.
    """
    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    while image.ndim > 2:
        image = image.mean(axis=0)
    if image.ndim == 0:
        image = np.full(spatial_shape, float(image), dtype=np.float32)
    elif image.shape != spatial_shape:
        image = cv2.resize(
            image,
            (int(spatial_shape[1]), int(spatial_shape[0])),
            interpolation=interpolation,
        )
    return image.astype(np.float32, copy=False)


def heatmap_overlay(
    base: np.ndarray,
    value_map: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a scalar map on the center pre-contrast slice."""
    value_map = resize_map_to_shape(value_map, base.shape[:2])
    return cv2.addWeighted(gray_to_bgr(base), 1.0 - alpha, heatmap(value_map, colormap), alpha, 0)


def heatmap_overlay_fixed(
    base: np.ndarray,
    value_map: np.ndarray,
    vmin: float,
    vmax: float,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a scalar map using a shared value range."""
    value_map = resize_map_to_shape(value_map, base.shape[:2])
    return cv2.addWeighted(
        gray_to_bgr(base),
        1.0 - alpha,
        heatmap_fixed(value_map, vmin, vmax, colormap),
        alpha,
        0,
    )


def gray_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image
    return cv2.cvtColor(normalize_to_uint8(image), cv2.COLOR_GRAY2BGR)


def add_title(panel: np.ndarray, title: str) -> np.ndarray:
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title[:80],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def resize_panel(panel: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(panel, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def make_grid(panels: Sequence[np.ndarray], cols: int = 4, fill: int = 0) -> np.ndarray:
    panels = [panel if panel.ndim == 3 else gray_to_bgr(panel) for panel in panels]
    max_h = max(panel.shape[0] for panel in panels)
    max_w = max(panel.shape[1] for panel in panels)
    normalized = []
    for panel in panels:
        canvas = np.full((max_h, max_w, 3), fill, dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        normalized.append(canvas)
    rows = []
    for start in range(0, len(normalized), cols):
        row = normalized[start : start + cols]
        while len(row) < cols:
            row.append(np.full((max_h, max_w, 3), fill, dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def overlay_mask(base: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.45) -> np.ndarray:
    output = gray_to_bgr(base)
    active = mask.astype(bool)
    color_img = np.zeros_like(output)
    color_img[:] = color
    output[active] = (
        (1.0 - alpha) * output[active].astype(np.float32)
        + alpha * color_img[active].astype(np.float32)
    ).astype(np.uint8)
    return output


def prediction_context_panels(sample: Dict, base: np.ndarray) -> List[np.ndarray]:
    """Return GT/prediction panels shared by attention and kinetic figures."""
    gt = (tensor_to_numpy(sample["mask"])[0] >= 0.5).astype(np.uint8)
    prob = tensor_to_numpy(sample["probability"])[0]
    pred = (prob >= float(sample["threshold"])).astype(np.uint8)
    merged = gray_to_bgr(base)
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    merged[gt_bool] = (0, 255, 0)
    merged[pred_bool] = (0, 0, 255)
    merged[np.logical_and(gt_bool, pred_bool)] = (0, 255, 255)
    return [
        add_title(gray_to_bgr(gt * 255), "GT mask"),
        add_title(gray_to_bgr(pred * 255), f"prediction t={float(sample['threshold']):g}"),
        add_title(heatmap(prob), "tumor probability"),
        add_title(merged, "GT green / Pred red / overlap yellow"),
    ]


def center_pre(image: torch.Tensor) -> np.ndarray:
    center = image.shape[0] // 2
    return image[center, 0].numpy()


def tensor_to_numpy(tensor_or_array) -> np.ndarray:
    if torch.is_tensor(tensor_or_array):
        return tensor_or_array.detach().float().cpu().numpy()
    return np.asarray(tensor_or_array)


def phase_labels(config: Dict, count: int) -> List[str]:
    labels = PHASE_LABELS_17[:count]
    if len(labels) < count:
        labels.extend([f"phase{i}" for i in range(len(labels), count)])
    return labels


def kinetic_labels(config: Dict, count: int) -> List[str]:
    model_cfg = config.get("model", {})
    map_names = list(model_cfg.get("kinetic_maps", []))
    phase_indices = model_cfg.get("phase_indices", {})
    n_dynamic = max(
        len(phase_indices.get("subtraction", []) or []),
        len(phase_indices.get("post", []) or []),
        1,
    )
    labels: List[str] = []
    for name in map_names:
        if name in {"sub_stack", "relative_enhancement"}:
            labels.extend([f"{name}_{i + 1}" for i in range(n_dynamic)])
        else:
            labels.append(str(name))
    if not labels:
        labels = [f"kinetic{i}" for i in range(count)]
    if len(labels) < count:
        labels.extend([f"kinetic{i}" for i in range(len(labels), count)])
    return labels[:count]


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


