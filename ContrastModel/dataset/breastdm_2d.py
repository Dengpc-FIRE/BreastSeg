from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def _normalize_channels(image: np.ndarray, mode: str = "zscore") -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return image
    out = image.copy()
    for c in range(out.shape[0]):
        plane = out[c]
        valid = plane[np.isfinite(plane)]
        if valid.size == 0:
            out[c] = 0
            continue
        if mode == "minmax":
            lo, hi = float(valid.min()), float(valid.max())
            out[c] = (plane - lo) / (hi - lo + 1e-6)
        else:
            nz = valid[np.abs(valid) > 1e-6]
            stats = nz if nz.size else valid
            out[c] = (plane - float(stats.mean())) / (float(stats.std()) + 1e-6)
    return out


def _load_mask(path: Path, size: Tuple[int, int] | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.float32)
        pil = Image.fromarray((mask * 255).astype(np.uint8))
    else:
        pil = Image.open(path).convert("L")
    if size is not None:
        pil = pil.resize((size[1], size[0]), Image.NEAREST)
    return (np.asarray(pil) > 0).astype(np.float32)


def _resize_image(image: np.ndarray, size: Tuple[int, int] | None) -> np.ndarray:
    if size is None:
        return image
    channels = []
    for plane in image:
        pil = Image.fromarray(plane.astype(np.float32))
        pil = pil.resize((size[1], size[0]), Image.BILINEAR)
        channels.append(np.asarray(pil, dtype=np.float32))
    return np.stack(channels, axis=0)


class BreastDM2DSlices(Dataset):
    """Reads processed_17ch_dce split folders as [17, H, W] image tensors."""

    def __init__(
        self,
        split_root: str | Path,
        image_size: Tuple[int, int] | None = None,
        normalize: str = "zscore",
        input_phase_indices: Sequence[int] | None = None,
    ) -> None:
        self.split_root = Path(split_root)
        self.data_dir = self.split_root / "data"
        self.mask_dir = self.split_root / "GT"
        self.image_size = tuple(image_size) if image_size is not None else None
        self.normalize = normalize
        self.input_phase_indices = (
            [int(index) for index in input_phase_indices]
            if input_phase_indices is not None
            else None
        )

        if not self.data_dir.exists():
            raise FileNotFoundError(f"2D data directory not found: {self.data_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"2D mask directory not found: {self.mask_dir}")

        self.samples: List[Tuple[Path, Path, str]] = []
        mask_by_stem: Dict[str, Path] = {}
        for ext in MASK_EXTENSIONS:
            mask_by_stem.update({p.stem: p for p in self.mask_dir.glob(f"*{ext}")})

        for image_path in sorted(self.data_dir.glob("*.npy")):
            mask_path = mask_by_stem.get(image_path.stem)
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
            raise ValueError(f"Expected 3D array for {image_path}, got shape {image.shape}")
        if image.shape[-1] == 17:
            image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] != 17:
            raise ValueError(f"Expected 17 channels in {image_path}, got shape {image.shape}")
        if self.input_phase_indices is not None:
            valid = [idx for idx in self.input_phase_indices if 0 <= idx < image.shape[0]]
            if len(valid) != len(self.input_phase_indices):
                raise ValueError(
                    f"Invalid input_phase_indices={self.input_phase_indices} "
                    f"for {image.shape[0]} channels in {image_path}"
                )
            image = image[valid]

        image = _resize_image(image, self.image_size)
        image = _normalize_channels(image, self.normalize)
        mask = _load_mask(mask_path, self.image_size)

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "id": sample_id,
        }
