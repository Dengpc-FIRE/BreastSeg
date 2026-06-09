from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def collect_split_pairs(split_path: str) -> List[Tuple[str, str]]:
    root = Path(split_path)
    if not root.exists():
        raise FileNotFoundError(f"split path not found: {root}")
    candidates = [
        (root / "data", root / "GT"),
        (root / "images", root / "masks"),
        (root / "image", root / "mask"),
    ]
    for image_root, mask_root in candidates:
        if image_root.exists() and mask_root.exists():
            return collect_pairs(str(image_root), str(mask_root))
    raise FileNotFoundError(
        f"No supported image/mask folders found under {root}. "
        "Expected data/GT, images/masks, or image/mask."
    )


def collect_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    if not image_root.exists():
        raise FileNotFoundError(f"image_dir not found: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"mask_dir not found: {mask_root}")
    pairs = []
    for image_path in sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        mask_path = _find_by_stem(mask_root, image_path.stem)
        if mask_path is not None:
            pairs.append((str(image_path), str(mask_path)))
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {image_root} and {mask_root}")
    return pairs


class BreastDM2DDataset(Dataset):
    """BreastDM 2D dataset for PDF-UNet adaptation.

    single_channel_pre uses channel 0 from processed DCE npy files.
    single_channel_sub uses channel 9 by default, corresponding to the first subtraction map.
    multi_phase uses the configured number of DCE channels without converting to 2.5D.
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        image_size: int = 128,
        input_mode: str = "single_channel_pre",
        in_channels: int = 1,
        mask_threshold: float = 0.0,
        use_center_slice_only: bool = True,
        sub_channel_index: int = 9,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = int(image_size)
        self.input_mode = input_mode
        self.in_channels = int(in_channels)
        self.mask_threshold = float(mask_threshold)
        self.use_center_slice_only = bool(use_center_slice_only)
        self.sub_channel_index = int(sub_channel_index)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = read_image(
            image_path,
            size=self.image_size,
            input_mode=self.input_mode,
            in_channels=self.in_channels,
            use_center_slice_only=self.use_center_slice_only,
            sub_channel_index=self.sub_channel_index,
        )
        mask = read_mask(mask_path, size=self.image_size, threshold=self.mask_threshold)
        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask[None, :, :]).float(),
            "image_path": image_path,
            "mask_path": mask_path,
        }


def read_image(
    path: str,
    size: int,
    input_mode: str,
    in_channels: int,
    use_center_slice_only: bool = True,
    sub_channel_index: int = 9,
) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        arr = np.load(str(path_obj))
        arr = _select_center_slice(arr) if use_center_slice_only else arr
        image = _select_channels(arr, input_mode, in_channels, sub_channel_index)
    else:
        gray = np.array(Image.open(path).convert("L"), dtype=np.float32)
        image = gray[None, :, :]
        if in_channels > 1:
            image = np.repeat(image, in_channels, axis=0)
    image = _resize_channels(image, size, cv2.INTER_LINEAR)
    return _minmax_per_channel(image)


def read_mask(path: str, size: int, threshold: float = 0.0) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        mask = np.load(str(path_obj))
        mask = _select_center_slice(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[-1] <= 64 else mask[mask.shape[0] // 2]
    else:
        mask = np.array(Image.open(path).convert("L"))
    mask = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > threshold).astype(np.float32)


def _select_center_slice(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        return arr
    center = arr.shape[0] // 2
    return arr[center]


def _select_channels(arr: np.ndarray, input_mode: str, in_channels: int, sub_channel_index: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        image = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[-1] <= 64:
        chw = np.transpose(arr, (2, 0, 1))
        if input_mode == "single_channel_pre":
            image = chw[0:1]
        elif input_mode == "single_channel_sub":
            idx = min(sub_channel_index, chw.shape[0] - 1)
            image = chw[idx:idx + 1]
        elif input_mode == "multi_phase":
            image = chw[:in_channels]
        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
    elif arr.ndim == 3:
        if input_mode == "multi_phase":
            image = arr[:in_channels]
        else:
            image = arr[0:1]
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}")
    if image.shape[0] < in_channels:
        pad = np.repeat(image[-1:, :, :], in_channels - image.shape[0], axis=0)
        image = np.concatenate([image, pad], axis=0)
    if image.shape[0] > in_channels:
        image = image[:in_channels]
    assert image.shape[0] == in_channels, f"Expected {in_channels} channels, got {image.shape[0]}"
    return image.astype(np.float32)


def _resize_channels(image: np.ndarray, size: int, interpolation) -> np.ndarray:
    return np.stack([cv2.resize(ch, (size, size), interpolation=interpolation) for ch in image], axis=0)


def _minmax_per_channel(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(np.float32)
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        ch = image[c]
        out[c] = (ch - ch.min()) / (ch.max() - ch.min() + eps)
    return out


def _find_by_stem(root: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None
