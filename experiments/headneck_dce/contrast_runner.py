from __future__ import annotations

import argparse
import csv
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRAST_ROOT = PROJECT_ROOT / "ContrastModel"
for _path in (PROJECT_ROOT, CONTRAST_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ContrastModel.dataset.config import as_tuple, load_config, select_device, set_seed  # noqa: E402
from ContrastModel.dataset.metrics import compute_case_metrics, summarize_metrics, write_metrics_csv, write_summary  # noqa: E402
from ContrastModel.dataset.models import align_logits, build_model, forward_model  # noqa: E402
from ContrastModel.dataset.training import dice_loss_with_logits, sliding_window_predict_3d  # noqa: E402


def _case_and_slice(sample_id: str) -> tuple[str, int]:
    match = re.match(r"^(?P<case>.+)_z(?P<z>\d+)$", str(sample_id))
    if not match:
        return str(sample_id), 0
    return match.group("case"), int(match.group("z"))


def _normalize_channels(image: np.ndarray, mode: str) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return image
    out = image.copy()
    for channel in range(out.shape[0]):
        values = out[channel]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[channel] = 0.0
            continue
        if mode == "minmax":
            lo, hi = float(finite.min()), float(finite.max())
            out[channel] = (values - lo) / (hi - lo + 1e-6)
        else:
            nonzero = finite[np.abs(finite) > 1e-6]
            stats = nonzero if nonzero.size else finite
            out[channel] = (values - float(stats.mean())) / (float(stats.std()) + 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _resize_image(image: np.ndarray, size: Tuple[int, int] | None) -> np.ndarray:
    if size is None or image.shape[-2:] == size:
        return image.astype(np.float32, copy=False)
    out = []
    for plane in image:
        pil = Image.fromarray(plane.astype(np.float32))
        out.append(np.asarray(pil.resize((size[1], size[0]), Image.BILINEAR), dtype=np.float32))
    return np.stack(out, axis=0)


def _load_mask(path: Path, size: Tuple[int, int] | None) -> np.ndarray:
    pil = Image.open(path).convert("L")
    if size is not None:
        pil = pil.resize((size[1], size[0]), Image.NEAREST)
    return (np.asarray(pil) > 0).astype(np.float32)


class HeadNeckDCE2DSlices(Dataset):
    """Reads HeadNeckDCE processed slices as arbitrary-channel tensors."""

    def __init__(self, split_path: str | Path, image_size: Tuple[int, int], input_channels: int, normalize: str) -> None:
        self.split_path = Path(split_path)
        self.data_dir = self.split_path / "data"
        self.mask_dir = self.split_path / "GT"
        self.image_size = tuple(image_size)
        self.input_channels = int(input_channels)
        self.normalize = normalize
        if not self.data_dir.exists():
            raise FileNotFoundError(f"HeadNeckDCE data directory not found: {self.data_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"HeadNeckDCE GT directory not found: {self.mask_dir}")
        self.samples = []
        masks = {p.stem: p for p in self.mask_dir.glob("*.png")}
        for image_path in sorted(self.data_dir.glob("*.npy")):
            mask_path = masks.get(image_path.stem)
            if mask_path is None:
                raise FileNotFoundError(f"Missing mask for {image_path.name} in {self.mask_dir}")
            self.samples.append((image_path, mask_path, image_path.stem))
        if not self.samples:
            raise FileNotFoundError(f"No .npy samples found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        image_path, mask_path, sample_id = self.samples[index]
        image = np.load(image_path)
        if image.ndim != 3:
            raise ValueError(f"Expected 3D array for {image_path}, got {image.shape}")
        if image.shape[0] == self.input_channels:
            image = image.astype(np.float32)
        elif image.shape[-1] == self.input_channels:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        else:
            raise ValueError(f"Expected {self.input_channels} channels in {image_path}, got {image.shape}")
        image = _resize_image(image, self.image_size)
        image = _normalize_channels(image, self.normalize)
        mask = _load_mask(mask_path, self.image_size)
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask[None]), "id": sample_id}


def _resize_volume_hw(volume: np.ndarray, size: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    if volume.shape[-2:] == size:
        return volume.astype(np.float32, copy=False)
    mode = Image.NEAREST if nearest else Image.BILINEAR
    if volume.ndim == 4:
        channels = [_resize_volume_hw(volume[c], size, nearest=nearest) for c in range(volume.shape[0])]
        return np.stack(channels, axis=0)
    out = []
    for plane in volume:
        pil = Image.fromarray(plane.astype(np.float32))
        out.append(np.asarray(pil.resize((size[1], size[0]), mode), dtype=np.float32))
    return np.stack(out, axis=0)


class HeadNeckDCE3DVolumes(Dataset):
    def __init__(self, split_path: str | Path, image_size: Tuple[int, int], input_channels: int, normalize: str) -> None:
        self.volume_dir = Path(split_path) / "volumes"
        self.image_size = tuple(image_size)
        self.input_channels = int(input_channels)
        self.normalize = normalize
        if not self.volume_dir.exists():
            raise FileNotFoundError(f"HeadNeckDCE volume directory not found: {self.volume_dir}")
        self.paths = sorted(self.volume_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz volumes found in {self.volume_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with np.load(path) as item:
            image = np.asarray(item["image"], dtype=np.float32)
            mask = np.asarray(item["mask"], dtype=np.float32)
        if image.shape[0] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels in {path}, got {image.shape}")
        if mask.ndim == 3:
            mask = mask[None]
        image = _resize_volume_hw(image, self.image_size, nearest=False)
        mask = _resize_volume_hw(mask, self.image_size, nearest=True)
        image = _normalize_channels(image, self.normalize)
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy((mask > 0).astype(np.float32)), "id": path.stem}


class HeadNeckDCE3DPatches(Dataset):
    def __init__(self, volumes: HeadNeckDCE3DVolumes, patch_size: Tuple[int, int, int], samples_per_volume: int, positive_crop_prob: float) -> None:
        self.volumes = volumes
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_volume = int(samples_per_volume)
        self.positive_crop_prob = float(positive_crop_prob)

    def __len__(self) -> int:
        return len(self.volumes) * self.samples_per_volume

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.volumes[index // self.samples_per_volume]
        image = sample["image"].numpy()
        mask = sample["mask"].numpy()
        image, mask = _crop_patch(image, mask, self.patch_size, self.positive_crop_prob)
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask), "id": str(sample["id"])}


def _crop_patch(image: np.ndarray, mask: np.ndarray, patch_size: Tuple[int, int, int], positive_prob: float) -> Tuple[np.ndarray, np.ndarray]:
    _, d, h, w = image.shape
    pd, ph, pw = patch_size
    nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
    if (d, h, w) != (nd, nh, nw):
        padded_image = np.zeros((image.shape[0], nd, nh, nw), dtype=image.dtype)
        padded_mask = np.zeros((1, nd, nh, nw), dtype=mask.dtype)
        padded_image[:, :d, :h, :w] = image
        padded_mask[:, :d, :h, :w] = mask
        image, mask = padded_image, padded_mask
    _, d, h, w = image.shape
    if np.random.rand() < positive_prob and mask.any():
        center = np.argwhere(mask[0] > 0)[np.random.randint(0, int(mask.sum()))]
    else:
        center = np.array([np.random.randint(0, d), np.random.randint(0, h), np.random.randint(0, w)])
    z = int(max(0, min(d - pd, center[0] - pd // 2)))
    y = int(max(0, min(h - ph, center[1] - ph // 2)))
    x = int(max(0, min(w - pw, center[2] - pw // 2)))
    return image[:, z : z + pd, y : y + ph, x : x + pw], mask[:, z : z + pd, y : y + ph, x : x + pw]


def make_dataset(cfg: Dict[str, Any], split: str, train: bool = False):
    data_cfg = cfg["data"]
    mode = str(data_cfg.get("mode", "2d")).lower()
    input_channels = int(cfg["model"]["input_channels"])
    image_size = as_tuple(data_cfg.get("image_size", [256, 256]), 2)
    normalize = str(data_cfg.get("normalize", "zscore"))
    if mode == "2d":
        return HeadNeckDCE2DSlices(data_cfg[f"{split}_path"], image_size, input_channels, normalize)
    volumes = HeadNeckDCE3DVolumes(data_cfg[f"{split}_path"], image_size, input_channels, normalize)
    if not train:
        return volumes
    return HeadNeckDCE3DPatches(
        volumes,
        patch_size=as_tuple(data_cfg.get("patch_size", [30, 128, 128]), 3),
        samples_per_volume=int(data_cfg.get("samples_per_volume", 4)),
        positive_crop_prob=float(data_cfg.get("positive_crop_prob", 0.7)),
    )


def make_loader(dataset, cfg: Dict[str, Any], shuffle: bool, batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or int(cfg["train"].get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    images = batch["image"].to(device=device, dtype=torch.float32)
    masks = batch["mask"].to(device=device, dtype=torch.float32)
    ids = batch["id"]
    if isinstance(ids, str):
        ids = [ids]
    return images, masks, list(ids)


def train_one_epoch(model, loader, optimizer, scaler, device, cfg) -> float:
    model.train()
    total = 0.0
    count = 0
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    for batch in tqdm(loader, desc="train", leave=False):
        images, masks, _ = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, extra = forward_model(model, images, None if getattr(model, "needs_target", False) else masks)
            logits = align_logits(logits, masks)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss_with_logits(logits, masks)
            if extra is not None and torch.isfinite(extra).all():
                loss = loss + float(cfg["train"].get("extra_loss_weight", 1.0)) * extra
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate_2d(model, loader, device, cfg) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    model.eval()
    cases: Dict[str, List[tuple[int, np.ndarray, np.ndarray]]] = {}
    threshold = float(cfg["eval"].get("threshold", 0.5))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    for batch in tqdm(loader, desc="eval", leave=False):
        images, masks, ids = _move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = forward_model(model, images, None if getattr(model, "needs_target", False) else masks)
            logits = align_logits(logits, masks)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = masks.detach().cpu().numpy()
        for idx, sample_id in enumerate(ids):
            case_id, slice_index = _case_and_slice(str(sample_id))
            cases.setdefault(case_id, []).append((slice_index, probs[idx, 0] >= threshold, targets[idx, 0] > 0))
    rows = []
    for case_id, slices in sorted(cases.items()):
        ordered = sorted(slices, key=lambda item: item[0])
        pred = np.stack([item[1] for item in ordered], axis=0)
        target = np.stack([item[2] for item in ordered], axis=0)
        metric = compute_case_metrics(pred, target)
        metric["id"] = case_id
        rows.append(metric)
    return summarize_metrics(rows), rows


@torch.no_grad()
def evaluate_3d(model, dataset, device, cfg) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    model.eval()
    rows = []
    threshold = float(cfg["eval"].get("threshold", 0.5))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    for sample in tqdm(dataset, desc="eval", leave=False):
        image = sample["image"][None].to(device=device, dtype=torch.float32)
        target = sample["mask"].numpy()[0] > 0
        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else nullcontext()
        with autocast_ctx:
            logits = sliding_window_predict_3d(model, image, cfg, device)
        pred = torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= threshold
        metric = compute_case_metrics(pred, target)
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


def train_model(model_dir: Path, model_key: str, cfg: Dict[str, Any], device: torch.device) -> None:
    set_seed(int(cfg["train"].get("seed", 2026)))
    mode = str(cfg["data"].get("mode", "2d")).lower()
    train_ds = make_dataset(cfg, "train", train=True)
    val_ds = make_dataset(cfg, "val", train=False)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = None if mode == "3d" else make_loader(val_ds, cfg, shuffle=False)
    model = build_model(model_key, cfg, model_dir).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 1e-4)), weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    best_score = -1.0
    best_epoch = 0
    patience = int(cfg["train"].get("early_stopping", 30))
    for epoch in range(1, int(cfg["train"].get("epochs", 100)) + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        summary, _ = evaluate_3d(model, val_ds, device, cfg) if mode == "3d" else evaluate_2d(model, val_loader, device, cfg)
        score = float(summary["mean_dice"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": cfg}, ckpt_dir / "latest_model.pth")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": cfg}, ckpt_dir / "best_model.pth")
        _append_log(Path(cfg["output"]["log_file"]), {"epoch": epoch, "train_loss": f"{loss:.6f}", "val_mean_dice": f"{score:.6f}", "val_mean_iou": f"{summary['mean_iou']:.6f}"})
        print(f"epoch={epoch} train_loss={loss:.6f} val_mean_dice={score:.6f} best={best_score:.6f}")
        if patience > 0 and epoch - best_epoch >= patience:
            print(f"early stopping at epoch {epoch}; best epoch {best_epoch}")
            break


def test_model(model_dir: Path, model_key: str, cfg: Dict[str, Any], device: torch.device, checkpoint: str | Path | None) -> Dict[str, float]:
    ckpt = Path(checkpoint) if checkpoint else Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = build_model(model_key, cfg, model_dir).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    mode = str(cfg["data"].get("mode", "2d")).lower()
    test_ds = make_dataset(cfg, "test", train=False)
    summary, rows = evaluate_3d(model, test_ds, device, cfg) if mode == "3d" else evaluate_2d(model, make_loader(test_ds, cfg, shuffle=False), device, cfg)
    out_dir = Path(cfg["output"]["test_dir"])
    write_metrics_csv(out_dir / "metrics.csv", rows)
    write_summary(out_dir / "summary.txt", summary)
    for key in ["mean_dice", "mean_iou", "mean_hd95", "mean_sensitivity", "mean_precision", "mean_accuracy"]:
        print(f"{key}: {summary[key]:.6f}")
    return summary


def parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    parser.add_argument("--amp", dest="amp", action="store_true")
    return parser.parse_args()


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
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
    args = parse_args(model_dir / "configs" / "headneck_dce_56ch.yaml")
    cfg = load_config(args.config, model_dir=model_dir, model_key=model_key)
    _apply_overrides(cfg, args)
    train_model(model_dir, model_key, cfg, select_device(args.device))


def run_test_cli(model_dir: str | Path, model_key: str) -> None:
    model_dir = Path(model_dir).resolve()
    args = parse_args(model_dir / "configs" / "headneck_dce_56ch.yaml")
    cfg = load_config(args.config, model_dir=model_dir, model_key=model_key)
    _apply_overrides(cfg, args)
    test_model(model_dir, model_key, cfg, select_device(args.device), args.checkpoint)
