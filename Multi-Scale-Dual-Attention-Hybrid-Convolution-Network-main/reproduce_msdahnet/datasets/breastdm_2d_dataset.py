from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def collect_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    if not image_root.exists():
        raise FileNotFoundError(f"image_dir not found: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"mask_dir not found: {mask_root}")

    pairs = []
    for image_path in sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        mask_path = _find_mask(mask_root, image_path.stem)
        if mask_path is not None:
            pairs.append((str(image_path), str(mask_path)))
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {image_root} and {mask_root}")
    return pairs


def collect_split_pairs(split_root: str) -> List[Tuple[str, str]]:
    root = Path(split_root)
    candidates = [
        (root / "pre_contrast_images", root / "masks"),
        (root / "data", root / "GT"),
        (root / "images", root / "masks"),
    ]
    for image_root, mask_root in candidates:
        if image_root.exists() and mask_root.exists():
            return collect_pairs(str(image_root), str(mask_root))
    raise FileNotFoundError(f"No supported image/mask folders found under split root: {root}")


def convert_processed_17ch_to_breastdm_dirs(source_root: str, output_root: str) -> dict:
    """Create reproduction-local pre-contrast image/mask folders from processed_17ch_dce.

    Each source data file is expected to be [H,W,17]. Channel 0 is used as
    pre-contrast input, matching the BreaDM table description.
    """
    source = Path(source_root)
    output = Path(output_root)
    image_out = output / "pre_contrast_images"
    mask_out = output / "masks"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "masks": 0, "skipped": 0}
    for split in ("train", "val", "test"):
        data_dir = source / split / "data"
        gt_dir = source / split / "GT"
        if not data_dir.exists() or not gt_dir.exists():
            continue
        for npy_path in sorted(data_dir.glob("*.npy")):
            arr = np.load(str(npy_path), mmap_mode="r")
            if arr.ndim != 3:
                stats["skipped"] += 1
                continue
            pre = _select_channel(arr, channel_index=0)
            pre = _minmax_uint8(pre)
            image_name = f"{split}_{npy_path.stem}.png"
            cv2.imwrite(str(image_out / image_name), pre)
            stats["images"] += 1

            mask_path = _find_mask(gt_dir, npy_path.stem)
            if mask_path is not None:
                mask = _read_mask(str(mask_path), size=None)
                cv2.imwrite(str(mask_out / image_name), (mask * 255).astype(np.uint8))
                stats["masks"] += 1
    return stats


def convert_processed_17ch_to_fixed_split_dirs(source_root: str, output_root: str, output_format: str = "npy") -> dict:
    """Create train/val/test pre-contrast folders from processed_17ch_dce."""
    source = Path(source_root)
    output = Path(output_root)
    output_format = output_format.lower()
    if output_format not in {"npy", "png"}:
        raise ValueError(f"Unsupported output_format: {output_format}")
    stats = {"images": 0, "masks": 0, "skipped": 0, "splits": {}}
    for split in ("train", "val", "test"):
        data_dir = source / split / "data"
        gt_dir = source / split / "GT"
        image_out = output / split / "pre_contrast_images"
        mask_out = output / split / "masks"
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        split_stats = {"images": 0, "masks": 0, "skipped": 0}
        if not data_dir.exists() or not gt_dir.exists():
            stats["splits"][split] = split_stats
            continue
        for npy_path in sorted(data_dir.glob("*.npy")):
            arr = np.load(str(npy_path), mmap_mode="r")
            if arr.ndim != 3:
                split_stats["skipped"] += 1
                stats["skipped"] += 1
                continue
            image_name = f"{npy_path.stem}.{output_format}"
            if output_format == "npy":
                np.save(str(image_out / image_name), np.asarray(arr, dtype=np.float32))
            else:
                pre = _select_channel(arr, channel_index=0)
                cv2.imwrite(str(image_out / image_name), _minmax_uint8(pre))
            split_stats["images"] += 1
            stats["images"] += 1

            mask_path = _find_mask(gt_dir, npy_path.stem)
            if mask_path is not None:
                mask = _read_mask(str(mask_path), size=None)
                if output_format == "npy":
                    np.save(str(mask_out / image_name), mask.astype(np.float32))
                else:
                    cv2.imwrite(str(mask_out / image_name), (mask * 255).astype(np.uint8))
                split_stats["masks"] += 1
                stats["masks"] += 1
        stats["splits"][split] = split_stats
    return stats


class BreastDM2DDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        image_size: int = 256,
        gray_to_rgb: bool = False,
        mask_threshold: float = 0.0,
        input_mode: str = "single_channel_pre",
        channel_index: int = None,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = int(image_size)
        self.gray_to_rgb = bool(gray_to_rgb)
        self.mask_threshold = float(mask_threshold)
        self.input_mode = input_mode
        self.channel_index = channel_index

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = _read_image(image_path, self.image_size, input_mode=self.input_mode, channel_index=self.channel_index)
        mask = _read_mask(mask_path, self.image_size, threshold=self.mask_threshold)

        image = image[None, :, :]
        if self.gray_to_rgb:
            image = np.repeat(image, 3, axis=0)
        mask = mask[None, :, :]
        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
            "image_path": image_path,
            "mask_path": mask_path,
        }


def _find_mask(mask_root: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"):
        candidate = mask_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_image(path: str, size: int, input_mode: str = "single_channel_pre", channel_index: int = None) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        arr = np.load(str(path_obj))
        if arr.ndim == 3:
            arr = _select_channel(arr, channel_index=_channel_index(input_mode, channel_index))
        image = np.asarray(arr, dtype=np.float32)
    else:
        image = np.array(Image.open(path).convert("L"), dtype=np.float32)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return _minmax_float(image)


def _channel_index(input_mode: str, channel_index: int = None) -> int:
    if channel_index is not None:
        return int(channel_index)
    if input_mode == "single_channel_pre":
        return 0
    if input_mode == "single_channel_post":
        return 1
    if input_mode == "single_channel_sub":
        return 9
    raise ValueError(f"Unsupported input_mode for single-channel MSDAHNet: {input_mode}")


def _read_mask(path: str, size: Optional[int], threshold: float = 0.0) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        mask = np.load(str(path_obj))
        if mask.ndim == 3:
            mask = _select_mask_plane(mask)
    else:
        mask = np.array(Image.open(path).convert("L"))
    if size is not None:
        mask = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > threshold).astype(np.float32)


def _minmax_float(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + eps)


def _minmax_uint8(image: np.ndarray) -> np.ndarray:
    image = _minmax_float(image)
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def _select_channel(arr: np.ndarray, channel_index: int = 0) -> np.ndarray:
    """Select one DCE channel from either HWC or CHW arrays."""
    if arr.ndim != 3:
        return np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] <= 64:
        idx = min(channel_index, arr.shape[-1] - 1)
        return np.asarray(arr[:, :, idx], dtype=np.float32)
    if arr.shape[0] <= 64:
        idx = min(channel_index, arr.shape[0] - 1)
        return np.asarray(arr[idx, :, :], dtype=np.float32)
    return np.asarray(arr[:, :, channel_index], dtype=np.float32)


def _select_mask_plane(mask: np.ndarray) -> np.ndarray:
    """Select a 2D mask from HWC, CHW, or slice-first arrays."""
    if mask.ndim != 3:
        return mask
    if mask.shape[-1] <= 4:
        return mask[:, :, 0]
    if mask.shape[0] <= 4:
        return mask[0, :, :]
    return mask[mask.shape[0] // 2, :, :]
