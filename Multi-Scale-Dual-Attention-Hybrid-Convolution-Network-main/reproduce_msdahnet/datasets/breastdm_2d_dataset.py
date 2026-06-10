from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
SEG_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def convert_seg_to_fixed_split_dirs(
    source_root: str,
    output_root: str,
    output_channels: int = 17,
    label_phase: str = "VIBRANT",
) -> dict:
    """Build fixed train/val/test samples directly from BreastSeg/seg.

    Output layout:
      split/data/*.npy: [H,W,C], C in {9,17}
      split/GT/*.npy:   [H,W] binary mask

    Channel layout for C=17:
      0=VIBRANT, 1-8=VIBRANT+C1..C8, 9-16=SUB1..SUB8
    Channel layout for C=9:
      0=VIBRANT, 1-8=SUB1..SUB8

    Labels default to labels/<patient>/VIBRANT to match the requested
    BreastDM baseline construction.
    """
    if int(output_channels) not in {9, 17}:
        raise ValueError(f"output_channels must be 9 or 17, got {output_channels}")
    source = Path(source_root)
    output = Path(output_root)
    stats = {"images": 0, "masks": 0, "skipped": 0, "splits": {}}
    for split in ("train", "val", "test"):
        images_root = source / split / "images"
        labels_root = source / split / "labels"
        out_data = output / split / "data"
        out_gt = output / split / "GT"
        out_data.mkdir(parents=True, exist_ok=True)
        out_gt.mkdir(parents=True, exist_ok=True)
        split_stats = {"images": 0, "masks": 0, "skipped": 0}
        if not images_root.exists() or not labels_root.exists():
            stats["splits"][split] = split_stats
            continue
        for patient_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
            patient_stats = _convert_seg_patient(
                patient_dir=patient_dir,
                label_patient_dir=labels_root / patient_dir.name,
                out_data=out_data,
                out_gt=out_gt,
                output_channels=int(output_channels),
                label_phase=label_phase,
            )
            for key in split_stats:
                split_stats[key] += patient_stats[key]
                stats[key] += patient_stats[key]
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

        if image.ndim == 2:
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
            arr = _select_input(arr, input_mode=input_mode, channel_index=channel_index)
        image = np.asarray(arr, dtype=np.float32)
    else:
        image = np.array(Image.open(path).convert("L"), dtype=np.float32)
    image = _resize_image(image, size)
    return _minmax_image(image)


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


def _select_input(arr: np.ndarray, input_mode: str, channel_index: int = None) -> np.ndarray:
    if input_mode in {"single_channel_pre", "single_channel_post", "single_channel_sub"}:
        return _select_channel(arr, channel_index=_channel_index(input_mode, channel_index))
    chw = _as_chw(arr)
    if input_mode == "multi_channel_9":
        if chw.shape[0] >= 17:
            return np.concatenate([chw[0:1], chw[9:17]], axis=0)
        return chw[:9]
    if input_mode == "multi_channel_17":
        if chw.shape[0] < 17:
            pad = np.repeat(chw[-1:], 17 - chw.shape[0], axis=0)
            return np.concatenate([chw, pad], axis=0)
        return chw[:17]
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def _as_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    if arr.shape[-1] <= 64:
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)
    if arr.shape[0] <= 64:
        return arr.astype(np.float32)
    raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 2:
        return cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    if image.ndim == 3:
        channels = [cv2.resize(ch.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR) for ch in image]
        return np.stack(channels, axis=0)
    raise ValueError(f"Unsupported image ndim: {image.ndim}")


def _minmax_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return _minmax_float(image)
    out = np.empty_like(image, dtype=np.float32)
    for idx in range(image.shape[0]):
        out[idx] = _minmax_float(image[idx])
    return out


def _select_mask_plane(mask: np.ndarray) -> np.ndarray:
    """Select a 2D mask from HWC, CHW, or slice-first arrays."""
    if mask.ndim != 3:
        return mask
    if mask.shape[-1] <= 4:
        return mask[:, :, 0]
    if mask.shape[0] <= 4:
        return mask[0, :, :]
    return mask[mask.shape[0] // 2, :, :]


def _convert_seg_patient(patient_dir: Path, label_patient_dir: Path, out_data: Path, out_gt: Path, output_channels: int, label_phase: str) -> dict:
    stats = {"images": 0, "masks": 0, "skipped": 0}
    pre_dir = patient_dir / "VIBRANT"
    label_dir = label_patient_dir / label_phase
    if not pre_dir.exists() or not label_dir.exists():
        return stats
    post_dirs = [patient_dir / f"VIBRANT+C{i}" for i in range(1, 9)]
    sub_dirs = [patient_dir / f"SUB{i}" for i in range(1, 9)]
    for slice_path in sorted(p for p in pre_dir.iterdir() if p.suffix.lower() in SEG_IMAGE_EXTENSIONS):
        pre = _read_seg_gray(slice_path)
        if pre is None:
            stats["skipped"] += 1
            continue
        channels = [pre]
        if output_channels == 17:
            channels.extend(_read_seg_or_zero(d / slice_path.name, pre.shape) for d in post_dirs)
        channels.extend(_read_seg_or_zero(d / slice_path.name, pre.shape) for d in sub_dirs)
        if len(channels) != output_channels:
            stats["skipped"] += 1
            continue
        label_path = _find_seg_label(label_dir, slice_path.stem)
        if label_path is None:
            stats["skipped"] += 1
            continue
        mask = _read_seg_mask(label_path, pre.shape)
        if mask is None:
            stats["skipped"] += 1
            continue
        stem = f"{patient_dir.name}_{slice_path.stem}"
        np.save(str(out_data / f"{stem}.npy"), np.stack(channels, axis=-1).astype(np.float32))
        np.save(str(out_gt / f"{stem}.npy"), mask.astype(np.float32))
        stats["images"] += 1
        stats["masks"] += 1
    return stats


def _read_seg_gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image.astype(np.float32)


def _read_seg_or_zero(path: Path, shape) -> np.ndarray:
    image = _read_seg_gray(path)
    if image is None:
        return np.zeros(shape, dtype=np.float32)
    return image


def _read_seg_mask(path: Path, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def _find_seg_label(label_dir: Path, slice_stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = label_dir / f"{slice_stem}{ext}"
        if candidate.exists():
            return candidate
    if not label_dir.exists():
        return None
    for path in label_dir.iterdir():
        if path.suffix.lower() in SEG_IMAGE_EXTENSIONS and path.stem in {slice_stem, f"{slice_stem}_mask"}:
            return path
    return None
