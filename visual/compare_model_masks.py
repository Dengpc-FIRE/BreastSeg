from __future__ import annotations

import argparse
import csv
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.common import center_pre, iter_outputs, normalize_to_uint8, overlay_mask


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def safe_filename(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def sample_id_from_name(name: str) -> str:
    return Path(str(name)).stem


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def binary_mask_from_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    probability = np.nan_to_num(np.asarray(probability, dtype=np.float32), nan=0.0)
    return (probability >= float(threshold)).astype(np.uint8)


def load_mask_file(path: Path, threshold: float = 0.5) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.squeeze(arr)
        return binary_mask_from_probability(arr, threshold)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask image: {path}")
    return (mask > 127).astype(np.uint8)


def write_overlay(
    output_root: Path,
    sample_id: str,
    model_label: str,
    base: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> Path:
    base_u8 = normalize_to_uint8(base)
    if mask.shape != base_u8.shape:
        mask = cv2.resize(mask.astype(np.uint8), (base_u8.shape[1], base_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = overlay_mask(base_u8, mask.astype(np.uint8), color=(0, 0, 255), alpha=alpha)
    sample_dir = output_root / safe_filename(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{safe_filename(model_label)}.png"
    cv2.imwrite(str(output_path), overlay)
    return output_path


def find_existing_mask(mask_dir: Path, sample_id: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = mask_dir / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(mask_dir.rglob(f"{sample_id}.*"))
    for match in matches:
        if match.suffix.lower() in IMAGE_EXTENSIONS and match.is_file():
            return match
    return None


def load_base_images(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    base_cfg = cfg.get("base", {})
    split_path = base_cfg.get("split_path")
    if not split_path:
        return {}
    split_path = Path(split_path)
    data_dir = split_path / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Base data directory not found: {data_dir}")

    mode = str(base_cfg.get("mode", "kpta25d")).lower()
    base_images: dict[str, np.ndarray] = {}
    for npy_path in sorted(data_dir.glob("*.npy")):
        arr = np.load(npy_path)
        if mode in {"kpta25d", "25d", "2.5d"}:
            if arr.ndim != 4:
                raise ValueError(f"Expected [K,T,H,W] base input, got {arr.shape} for {npy_path}")
            base = arr[arr.shape[0] // 2, 0]
        elif mode in {"contrast2d", "2d", "17ch"}:
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D base input, got {arr.shape} for {npy_path}")
            base = arr[0] if arr.shape[0] in {1, 3, 9, 17} else arr[..., 0]
        else:
            raise ValueError(f"Unknown base.mode={mode!r}")
        base_images[npy_path.stem] = base.astype(np.float32)
    return base_images


def run_kpta25d_model(
    model_cfg: dict[str, Any],
    output_root: Path,
    alpha: float,
    written: list[dict[str, str]],
) -> None:
    label = str(model_cfg["name"])
    args = Namespace(
        config=model_cfg["config"],
        checkpoint=model_cfg.get("checkpoint"),
        split=model_cfg.get("split", "test"),
        split_path=model_cfg.get("split_path"),
        output_dir=None,
        output_subdir="mask_comparison_tmp",
        batch_size=int(model_cfg.get("batch_size", 4)),
        num_workers=int(model_cfg.get("num_workers", 4)),
        max_samples=int(model_cfg.get("max_samples", 1_000_000)),
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    for sample in iter_outputs(args, f"{label} masks"):
        sample_id = sample_id_from_name(sample["name"])
        base = center_pre(sample["image"])
        probability = sample["probability"][0].numpy()
        pred = binary_mask_from_probability(probability, sample["threshold"])
        path = write_overlay(output_root, sample_id, label, base, pred, alpha)
        written.append({"sample_id": sample_id, "model": label, "path": str(path)})


def run_contrast2d_model(
    model_cfg: dict[str, Any],
    output_root: Path,
    alpha: float,
    written: list[dict[str, str]],
) -> None:
    from ContrastModel.dataset.config import load_config as load_contrast_config
    from ContrastModel.dataset.training import (
        _load_model_for_test,
        _move_batch,
        make_dataset,
        make_loader,
    )
    from ContrastModel.dataset.models import align_logits, forward_model
    from inference.whole_breast_constraint import build_whole_breast_constraint

    label = str(model_cfg["name"])
    model_dir = Path(model_cfg["model_dir"])
    model_key = str(model_cfg["model_key"])
    cfg = load_contrast_config(model_cfg["config"], model_dir=model_dir, model_key=model_key)
    if model_cfg.get("split_path"):
        cfg["data"][f"{model_cfg.get('split', 'test')}_path"] = str(Path(model_cfg["split_path"]).resolve())
    if "threshold" in model_cfg:
        cfg["eval"]["threshold"] = float(model_cfg["threshold"])
    if "batch_size" in model_cfg:
        cfg["train"]["batch_size"] = int(model_cfg["batch_size"])
    if "num_workers" in model_cfg:
        cfg["train"]["num_workers"] = int(model_cfg["num_workers"])

    mode = str(cfg["data"].get("mode", "2d")).lower()
    if mode != "2d":
        raise ValueError(
            f"{label} uses data.mode={mode!r}. This comparison script supports direct "
            "inference for 2D contrast models only. Generate masks first and use mode: mask_dir."
        )

    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = model_cfg.get("checkpoint") or (Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth")
    model = _load_model_for_test(model_dir, model_key, cfg, checkpoint, device)
    model.eval()

    split = str(model_cfg.get("split", "test"))
    dataset = make_dataset(cfg, split, train=False)
    loader = make_loader(dataset, cfg, shuffle=False, batch_size=int(model_cfg.get("batch_size", cfg["train"].get("batch_size", 4))))
    threshold = float(cfg["eval"].get("threshold", 0.5))
    whole_breast = build_whole_breast_constraint(
        cfg,
        device=device,
        output_path=Path(cfg["output"]["test_dir"]).parent,
    )

    use_amp = bool(model_cfg.get("amp", cfg["train"].get("amp", True))) and device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{label} masks"):
            images, masks, ids = _move_batch(batch, device)
            model_target = None if getattr(model, "needs_target", False) else masks
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = forward_model(model, images, model_target)
                logits = align_logits(logits, masks)
                probabilities = torch.sigmoid(logits)
            if whole_breast is not None:
                probabilities, _ = whole_breast.constrain_probabilities(probabilities, ids, dataset)
            probs = probabilities.detach().float().cpu().numpy()[:, 0]
            images_cpu = images.detach().float().cpu().numpy()
            for probability, image, sample_id in zip(probs, images_cpu, ids):
                pred = binary_mask_from_probability(probability, threshold)
                base = image[0]
                path = write_overlay(output_root, str(sample_id), label, base, pred, alpha)
                written.append({"sample_id": str(sample_id), "model": label, "path": str(path)})


def run_mask_dir_model(
    model_cfg: dict[str, Any],
    output_root: Path,
    base_images: dict[str, np.ndarray],
    alpha: float,
    written: list[dict[str, str]],
) -> None:
    label = str(model_cfg["name"])
    mask_dir = Path(model_cfg["mask_dir"])
    if not mask_dir.is_dir():
        if bool(model_cfg.get("skip_missing", True)):
            print(f"[warning] skip {label}: mask directory not found: {mask_dir}")
            return
        raise FileNotFoundError(f"Mask directory not found for {label}: {mask_dir}")
    threshold = float(model_cfg.get("threshold", 0.5))
    if not base_images:
        raise ValueError("mode: mask_dir requires global base.split_path so overlays can be drawn on pre images.")

    for sample_id, base in tqdm(sorted(base_images.items()), desc=f"{label} masks"):
        mask_path = find_existing_mask(mask_dir, sample_id)
        if mask_path is None:
            continue
        pred = load_mask_file(mask_path, threshold=threshold)
        path = write_overlay(output_root, sample_id, label, base, pred, alpha)
        written.append({"sample_id": sample_id, "model": label, "path": str(path)})


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "model", "path"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-slice red-mask comparison folders for multiple models.")
    parser.add_argument("--models_config", default="visual/model_mask_comparison.yaml")
    parser.add_argument("--output", default=None, help="Override output directory from config.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected model names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.models_config)
    cfg = read_yaml(cfg_path)
    output_root = Path(args.output or cfg.get("output_dir", "visual/test/masks"))
    output_root.mkdir(parents=True, exist_ok=True)
    alpha = float(cfg.get("overlay_alpha", 0.55))
    base_images = load_base_images(cfg)
    selected = set(args.only or [])
    written: list[dict[str, str]] = []

    for model_cfg in cfg.get("models", []):
        if model_cfg.get("enabled", True) is False:
            continue
        name = str(model_cfg.get("name", "model"))
        if selected and name not in selected:
            continue
        mode = str(model_cfg.get("mode", "kpta25d")).lower()
        if mode in {"kpta", "kpta25d", "spta", "spta_net"}:
            run_kpta25d_model(model_cfg, output_root, alpha, written)
        elif mode in {"contrast2d", "contrast", "2d"}:
            run_contrast2d_model(model_cfg, output_root, alpha, written)
        elif mode in {"mask_dir", "predictions", "existing"}:
            run_mask_dir_model(model_cfg, output_root, base_images, alpha, written)
        else:
            raise ValueError(f"Unknown model mode for {name}: {mode}")

    write_manifest(output_root / "manifest.csv", written)
    print(f"Saved {len(written)} overlays to {output_root}")


if __name__ == "__main__":
    main()
