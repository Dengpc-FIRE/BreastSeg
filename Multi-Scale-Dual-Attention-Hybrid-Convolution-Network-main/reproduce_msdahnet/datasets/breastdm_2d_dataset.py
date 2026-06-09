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
            pre = np.asarray(arr[:, :, 0], dtype=np.float32)
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


class BreastDM2DDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        image_size: int = 256,
        gray_to_rgb: bool = False,
        mask_threshold: float = 0.0,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = int(image_size)
        self.gray_to_rgb = bool(gray_to_rgb)
        self.mask_threshold = float(mask_threshold)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = _read_image(image_path, self.image_size)
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


def _read_image(path: str, size: int) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        arr = np.load(str(path_obj))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        image = np.asarray(arr, dtype=np.float32)
    else:
        image = np.array(Image.open(path).convert("L"), dtype=np.float32)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return _minmax_float(image)


def _read_mask(path: str, size: Optional[int], threshold: float = 0.0) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        mask = np.load(str(path_obj))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
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
