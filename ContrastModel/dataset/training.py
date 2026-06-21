from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .breastdm_2d import BreastDM2DSlices
from .breastdm_3d import BreastDM3DPatches, BreastDM3DVolumes
from .config import as_tuple, load_config, select_device, set_seed
from .metrics import compute_case_metrics, summarize_metrics, write_metrics_csv, write_summary
from .models import align_logits, build_model, forward_model


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    den = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, extra_loss: torch.Tensor | None, extra_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_with_logits(logits, target)
    loss = bce + dice
    if extra_loss is not None and extra_weight > 0 and torch.isfinite(extra_loss).all():
        loss = loss + float(extra_weight) * extra_loss
    return loss


def make_dataset(cfg: Dict[str, Any], split: str, train: bool = False):
    data_cfg = cfg["data"]
    mode = str(data_cfg.get("mode", "2d")).lower()
    if mode == "2d":
        path = data_cfg[f"{split}_path"]
        return BreastDM2DSlices(
            path,
            image_size=as_tuple(data_cfg.get("image_size", [256, 256]), 2),
            normalize=data_cfg.get("normalize", "zscore"),
        )

    volumes = BreastDM3DVolumes(
        raw_dataset_root=data_cfg["raw_dataset_root"],
        split=split,
        cache_root=data_cfg["cache_root"],
        phase_names=data_cfg["phase_names"],
        label_phase=data_cfg.get("label_phase"),
        normalize=data_cfg.get("normalize", "zscore"),
        allow_missing_phases=bool(data_cfg.get("allow_missing_phases", False)),
    )
    if train:
        return BreastDM3DPatches(
            volumes,
            patch_size=as_tuple(data_cfg.get("patch_size", [48, 128, 128]), 3),
            samples_per_volume=int(data_cfg.get("samples_per_volume", 4)),
            positive_crop_prob=float(data_cfg.get("positive_crop_prob", 0.7)),
        )
    return volumes


def make_loader(dataset, cfg: Dict[str, Any], shuffle: bool, batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or int(cfg["train"].get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    images = batch["image"].to(device=device, dtype=torch.float32)
    masks = batch["mask"].to(device=device, dtype=torch.float32)
    ids = batch["id"]
    if isinstance(ids, str):
        ids = [ids]
    return images, masks, list(ids)


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_score: float, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def _load_training_state(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"], strict=False)
    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])


def train_one_epoch(model, loader, optimizer, scaler, device, cfg) -> float:
    model.train()
    total = 0.0
    count = 0
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    extra_weight = float(cfg["train"].get("extra_loss_weight", 1.0))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 0.0) or 0.0)
    for batch in loader:
        images, masks, _ = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, extra = forward_model(model, images, masks)
            logits = align_logits(logits, masks)
            loss = segmentation_loss(logits, masks, extra, extra_weight)
        if not torch.isfinite(loss).all():
            finite_logits = torch.isfinite(logits)
            finite_masks = torch.isfinite(masks)
            if finite_logits.any():
                logit_min = float(logits[finite_logits].detach().min().cpu())
                logit_max = float(logits[finite_logits].detach().max().cpu())
            else:
                logit_min = float("nan")
                logit_max = float("nan")
            print(
                "warning: skip non-finite loss batch "
                f"loss={float(loss.detach().cpu()) if loss.numel() == 1 else 'non-scalar'} "
                f"logits_finite={bool(finite_logits.all())} logits_range=({logit_min:.4g},{logit_max:.4g}) "
                f"masks_finite={bool(finite_masks.all())}"
            )
            continue
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate_loader(model, loader, device, cfg) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    model.eval()
    threshold = float(cfg["eval"].get("threshold", 0.5))
    rows: List[Dict[str, float]] = []
    for batch in loader:
        images, masks, ids = _move_batch(batch, device)
        model_target = None if getattr(model, "needs_target", False) else masks
        logits, _ = forward_model(model, images, model_target)
        logits = align_logits(logits, masks)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = masks.detach().cpu().numpy()
        for i, sample_id in enumerate(ids):
            pred = probs[i, 0] >= threshold
            target = targets[i, 0] > 0
            metric = compute_case_metrics(pred, target)
            metric["id"] = sample_id
            rows.append(metric)
    return summarize_metrics(rows), rows


def _compute_starts(size: int, patch: int, overlap: float) -> List[int]:
    if size <= patch:
        return [0]
    stride = max(1, int(patch * (1.0 - overlap)))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _pad_3d_tensor(image: torch.Tensor, patch_size: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _, _, d, h, w = image.shape
    pd, ph, pw = patch_size
    nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
    if (d, h, w) == (nd, nh, nw):
        return image, (d, h, w)
    padded = torch.zeros((image.shape[0], image.shape[1], nd, nh, nw), dtype=image.dtype, device=image.device)
    padded[:, :, :d, :h, :w] = image
    return padded, (d, h, w)


@torch.no_grad()
def sliding_window_predict_3d(model, image: torch.Tensor, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    patch_size = as_tuple(cfg["data"].get("patch_size", [48, 128, 128]), 3)
    overlap = float(cfg["eval"].get("sliding_window_overlap", 0.5))
    image, original_shape = _pad_3d_tensor(image, patch_size)
    _, _, d, h, w = image.shape
    pd, ph, pw = patch_size
    output = torch.zeros((1, 1, d, h, w), dtype=torch.float32, device=device)
    count = torch.zeros_like(output)
    for z in _compute_starts(d, pd, overlap):
        for y in _compute_starts(h, ph, overlap):
            for x in _compute_starts(w, pw, overlap):
                patch = image[:, :, z : z + pd, y : y + ph, x : x + pw]
                logits, _ = forward_model(model, patch, None)
                logits = align_logits(logits, torch.zeros((1, 1, pd, ph, pw), device=device))
                output[:, :, z : z + pd, y : y + ph, x : x + pw] += logits
                count[:, :, z : z + pd, y : y + ph, x : x + pw] += 1
    output = output / count.clamp_min(1.0)
    od, oh, ow = original_shape
    return output[:, :, :od, :oh, :ow]


@torch.no_grad()
def evaluate_3d(model, dataset, device, cfg) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    model.eval()
    threshold = float(cfg["eval"].get("threshold", 0.5))
    rows: List[Dict[str, float]] = []
    for sample in dataset:
        image = sample["image"][None].to(device=device, dtype=torch.float32)
        mask = sample["mask"].numpy()[0] > 0
        logits = sliding_window_predict_3d(model, image, cfg, device)
        pred = torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= threshold
        metric = compute_case_metrics(pred, mask)
        metric["id"] = str(sample["id"])
        rows.append(metric)
    return summarize_metrics(rows), rows


def _append_log(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_model(model_dir: str | Path, model_key: str, cfg: Dict[str, Any], device: torch.device) -> None:
    set_seed(int(cfg["train"].get("seed", 2026)))
    train_ds = make_dataset(cfg, "train", train=True)
    val_ds = make_dataset(cfg, "val", train=False)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    mode = str(cfg["data"].get("mode", "2d")).lower()
    val_loader = None if mode == "3d" else make_loader(val_ds, cfg, shuffle=False)

    model = build_model(model_key, cfg, model_dir).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1.0e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1.0e-5)),
    )
    scheduler = None
    if str(cfg["train"].get("scheduler", "")).lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    log_path = Path(cfg["output"]["log_file"])
    best_score = -1.0
    best_epoch = 0
    patience = int(cfg["train"].get("early_stopping", 30))
    collapse_guard = bool(cfg["train"].get("collapse_guard", False))
    collapse_ratio = float(cfg["train"].get("collapse_ratio", 0.25))
    collapse_min_epoch = int(cfg["train"].get("collapse_min_epoch", 5))
    collapse_min_best = float(cfg["train"].get("collapse_min_best", 0.05))
    collapse_lr_factor = float(cfg["train"].get("collapse_lr_factor", 0.2))
    collapse_stop_after = int(cfg["train"].get("collapse_stop_after", 0))
    collapse_events = 0
    for epoch in range(1, int(cfg["train"].get("epochs", 100)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        if mode == "3d":
            val_summary, _ = evaluate_3d(model, val_ds, device, cfg)
        else:
            val_summary, _ = evaluate_loader(model, val_loader, device, cfg)
        score = float(val_summary["mean_dice"])
        if scheduler is not None:
            scheduler.step(score)
        _save_checkpoint(ckpt_dir / "latest_model.pth", model, optimizer, epoch, best_score, cfg)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            _save_checkpoint(ckpt_dir / "best_model.pth", model, optimizer, epoch, best_score, cfg)
        collapsed = (
            collapse_guard
            and epoch >= collapse_min_epoch
            and best_score >= collapse_min_best
            and score < best_score * collapse_ratio
        )
        _append_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_mean_dice": f"{val_summary['mean_dice']:.6f}",
                "val_mean_iou": f"{val_summary['mean_iou']:.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
                "collapse_guard": int(collapsed),
            },
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_mean_dice={score:.6f} "
            f"best={best_score:.6f} lr={optimizer.param_groups[0]['lr']:.8f}"
        )
        if collapsed:
            collapse_events += 1
            best_path = ckpt_dir / "best_model.pth"
            if best_path.exists():
                _load_training_state(best_path, model, optimizer, device)
            for group in optimizer.param_groups:
                group["lr"] = max(float(group["lr"]) * collapse_lr_factor, 1.0e-8)
            print(
                "collapse guard triggered: "
                f"epoch={epoch} score={score:.6f} best={best_score:.6f}; "
                f"restored best epoch {best_epoch}, lr={optimizer.param_groups[0]['lr']:.8f}"
            )
            if collapse_stop_after > 0 and collapse_events >= collapse_stop_after:
                print(f"stopping after {collapse_events} collapse guard event(s)")
                break
        if patience > 0 and epoch - best_epoch >= patience:
            print(f"early stopping at epoch {epoch}; best epoch {best_epoch}")
            break


def _load_model_for_test(model_dir: str | Path, model_key: str, cfg: Dict[str, Any], checkpoint: str | Path, device: torch.device):
    model = build_model(model_key, cfg, model_dir).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    return model


def _save_2d_predictions(rows: List[Dict[str, float]], loader, model, device, cfg, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(cfg["eval"].get("threshold", 0.5))
    with torch.no_grad():
        for batch in loader:
            images, masks, ids = _move_batch(batch, device)
            model_target = None if getattr(model, "needs_target", False) else masks
            logits, _ = forward_model(model, images, model_target)
            logits = align_logits(logits, masks)
            preds = (torch.sigmoid(logits).detach().cpu().numpy()[:, 0] >= threshold).astype(np.uint8) * 255
            for pred, sample_id in zip(preds, ids):
                Image.fromarray(pred).save(out_dir / f"{sample_id}.png")


def test_model(model_dir: str | Path, model_key: str, cfg: Dict[str, Any], device: torch.device, checkpoint: str | Path | None = None) -> Dict[str, float]:
    ckpt = Path(checkpoint) if checkpoint else Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = _load_model_for_test(model_dir, model_key, cfg, ckpt, device)
    mode = str(cfg["data"].get("mode", "2d")).lower()
    test_ds = make_dataset(cfg, "test", train=False)
    if mode == "3d":
        summary, rows = evaluate_3d(model, test_ds, device, cfg)
    else:
        loader = make_loader(test_ds, cfg, shuffle=False)
        summary, rows = evaluate_loader(model, loader, device, cfg)
        if bool(cfg["eval"].get("save_predictions", False)):
            _save_2d_predictions(rows, loader, model, device, cfg, Path(cfg["output"]["test_dir"]) / "predictions")

    test_dir = Path(cfg["output"]["test_dir"])
    write_metrics_csv(test_dir / "metrics.csv", rows)
    write_summary(test_dir / "summary.txt", summary)
    for key in ["mean_dice", "mean_iou", "mean_hd95", "mean_sensitivity", "mean_precision", "mean_accuracy"]:
        print(f"{key}: {summary[key]:.6f}")
    return summary


def _parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.amp is not None:
        cfg["train"]["amp"] = args.amp


def run_train_cli(model_dir: str | Path, model_key: str) -> None:
    model_dir = Path(model_dir).resolve()
    args = _parse_args(model_dir / "configs" / "breastdm_17ch.yaml")
    cfg = load_config(args.config, model_dir=model_dir, model_key=model_key)
    _apply_cli_overrides(cfg, args)
    device = select_device(args.device)
    train_model(model_dir, model_key, cfg, device)


def run_test_cli(model_dir: str | Path, model_key: str) -> None:
    model_dir = Path(model_dir).resolve()
    args = _parse_args(model_dir / "configs" / "breastdm_17ch.yaml")
    cfg = load_config(args.config, model_dir=model_dir, model_key=model_key)
    _apply_cli_overrides(cfg, args)
    device = select_device(args.device)
    test_model(model_dir, model_key, cfg, device, args.checkpoint)
