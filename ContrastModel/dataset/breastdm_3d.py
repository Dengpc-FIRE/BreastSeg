from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def _read_image_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return np.squeeze(arr).astype(np.float32)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def _read_stack(folder: Path) -> np.ndarray:
    if folder.is_file() and folder.suffix.lower() == ".npy":
        arr = np.load(folder)
        if arr.ndim == 2:
            arr = arr[None]
        return np.squeeze(arr).astype(np.float32)
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No image slices found in {folder}")
    slices = [_read_image_file(path) for path in files]
    return np.stack(slices, axis=0).astype(np.float32)


def _resize_volume(volume: np.ndarray, shape: Tuple[int, int, int], nearest: bool = False) -> np.ndarray:
    d, h, w = shape
    out = np.zeros((d, h, w), dtype=np.float32)
    mode = Image.NEAREST if nearest else Image.BILINEAR
    copy_d = min(d, volume.shape[0])
    for z in range(copy_d):
        plane = Image.fromarray(volume[z].astype(np.float32))
        out[z] = np.asarray(plane.resize((w, h), mode), dtype=np.float32)
    return out


def _normalize(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image.astype(np.float32)
    out = image.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        vals = out[c]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            out[c] = 0
            continue
        if mode == "minmax":
            lo, hi = float(finite.min()), float(finite.max())
            out[c] = (vals - lo) / (hi - lo + 1e-6)
        else:
            nz = finite[np.abs(finite) > 1e-6]
            stats = nz if nz.size else finite
            out[c] = (vals - float(stats.mean())) / (float(stats.std()) + 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _find_label_path(label_patient_dir: Path, label_phase: str | None, phase_names: Sequence[str]) -> Path:
    if label_phase:
        candidate = label_patient_dir / label_phase
        if candidate.exists():
            return candidate
    for name in ["GT", "label", "labels", "mask", "tumor"] + list(phase_names):
        candidate = label_patient_dir / name
        if candidate.exists():
            return candidate
    if label_patient_dir.exists():
        for ext in IMAGE_EXTENSIONS:
            if any(label_patient_dir.glob(f"*{ext}")):
                return label_patient_dir
        children = sorted([p for p in label_patient_dir.iterdir() if p.is_dir() or p.suffix.lower() == ".npy"])
        if children:
            return children[0]
    raise FileNotFoundError(f"No label phase found under {label_patient_dir}")


def build_or_load_volume(
    raw_dataset_root: str | Path,
    split: str,
    patient_id: str,
    cache_root: str | Path,
    phase_names: Sequence[str],
    label_phase: str | None = None,
    normalize: str = "zscore",
    allow_missing_phases: bool = False,
) -> Dict[str, np.ndarray | str]:
    raw_dataset_root = Path(raw_dataset_root)
    cache_path = Path(cache_root) / split / f"{patient_id}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = np.load(cache_path)
        return {"image": cached["image"], "mask": cached["mask"], "id": patient_id}

    image_patient_dir = raw_dataset_root / split / "images" / patient_id
    label_patient_dir = raw_dataset_root / split / "labels" / patient_id
    if not image_patient_dir.exists():
        raise FileNotFoundError(f"Patient image directory not found: {image_patient_dir}")

    stacks: List[np.ndarray] = []
    target_shape: Tuple[int, int, int] | None = None
    for phase in phase_names:
        phase_dir = image_patient_dir / phase
        if not phase_dir.exists():
            if not allow_missing_phases:
                raise FileNotFoundError(f"Missing phase {phase} for patient {patient_id}: {phase_dir}")
            if target_shape is None:
                raise FileNotFoundError(f"First phase {phase} is missing, cannot infer volume shape")
            stacks.append(np.zeros(target_shape, dtype=np.float32))
            continue
        stack = _read_stack(phase_dir)
        if target_shape is None:
            target_shape = stack.shape
        if stack.shape != target_shape:
            stack = _resize_volume(stack, target_shape)
        stacks.append(stack.astype(np.float32))

    if target_shape is None:
        raise FileNotFoundError(f"No phase volumes found for patient {patient_id}")

    label_path = _find_label_path(label_patient_dir, label_phase, phase_names)
    mask = _read_stack(label_path)
    if mask.shape != target_shape:
        mask = _resize_volume(mask, target_shape, nearest=True)
    mask = (mask > 0).astype(np.float32)[None]
    image = _normalize(np.stack(stacks, axis=0), normalize)

    np.savez_compressed(cache_path, image=image.astype(np.float32), mask=mask.astype(np.float32))
    return {"image": image.astype(np.float32), "mask": mask.astype(np.float32), "id": patient_id}


class BreastDM3DVolumes(Dataset):
    """Builds [17, D, H, W] patient volumes from raw BreastDM patient folders."""

    def __init__(
        self,
        raw_dataset_root: str | Path,
        split: str,
        cache_root: str | Path,
        phase_names: Sequence[str],
        label_phase: str | None = None,
        normalize: str = "zscore",
        allow_missing_phases: bool = False,
    ) -> None:
        self.raw_dataset_root = Path(raw_dataset_root)
        self.split = split
        self.cache_root = Path(cache_root)
        self.phase_names = list(phase_names)
        self.label_phase = label_phase
        self.normalize = normalize
        self.allow_missing_phases = allow_missing_phases
        images_root = self.raw_dataset_root / split / "images"
        if not images_root.exists():
            raise FileNotFoundError(f"3D images split directory not found: {images_root}")
        self.patient_ids = sorted([p.name for p in images_root.iterdir() if p.is_dir()])
        if not self.patient_ids:
            raise FileNotFoundError(f"No patient directories found in {images_root}")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        patient_id = self.patient_ids[index]
        item = build_or_load_volume(
            self.raw_dataset_root,
            self.split,
            patient_id,
            self.cache_root,
            self.phase_names,
            self.label_phase,
            self.normalize,
            self.allow_missing_phases,
        )
        return {
            "image": torch.from_numpy(item["image"]),
            "mask": torch.from_numpy(item["mask"]),
            "id": str(item["id"]),
        }


def _pad_to_patch(image: np.ndarray, mask: np.ndarray, patch_size: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    _, d, h, w = image.shape
    pd, ph, pw = patch_size
    new_shape = (max(d, pd), max(h, ph), max(w, pw))
    if new_shape == (d, h, w):
        return image, mask
    padded_image = np.zeros((image.shape[0], *new_shape), dtype=image.dtype)
    padded_mask = np.zeros((1, *new_shape), dtype=mask.dtype)
    padded_image[:, :d, :h, :w] = image
    padded_mask[:, :d, :h, :w] = mask
    return padded_image, padded_mask


def _crop_patch(image: np.ndarray, mask: np.ndarray, patch_size: Tuple[int, int, int], positive_prob: float) -> Tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_patch(image, mask, patch_size)
    _, d, h, w = image.shape
    pd, ph, pw = patch_size
    if np.random.rand() < positive_prob and mask.any():
        coords = np.argwhere(mask[0] > 0)
        center = coords[np.random.randint(0, len(coords))]
    else:
        center = np.array([np.random.randint(0, d), np.random.randint(0, h), np.random.randint(0, w)])

    starts = []
    for c, dim, patch in zip(center, (d, h, w), patch_size):
        lo = int(max(0, min(dim - patch, c - patch // 2)))
        starts.append(lo)
    z, y, x = starts
    return image[:, z : z + pd, y : y + ph, x : x + pw], mask[:, z : z + pd, y : y + ph, x : x + pw]


class BreastDM3DPatches(Dataset):
    def __init__(
        self,
        volume_dataset: BreastDM3DVolumes,
        patch_size: Tuple[int, int, int],
        samples_per_volume: int = 4,
        positive_crop_prob: float = 0.7,
    ) -> None:
        self.volume_dataset = volume_dataset
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_volume = int(samples_per_volume)
        self.positive_crop_prob = float(positive_crop_prob)

    def __len__(self) -> int:
        return len(self.volume_dataset) * self.samples_per_volume

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        volume_index = index // self.samples_per_volume
        sample = self.volume_dataset[volume_index]
        image = sample["image"].numpy()
        mask = sample["mask"].numpy()
        image_patch, mask_patch = _crop_patch(image, mask, self.patch_size, self.positive_crop_prob)
        return {
            "image": torch.from_numpy(image_patch.astype(np.float32)),
            "mask": torch.from_numpy(mask_patch.astype(np.float32)),
            "id": str(sample["id"]),
        }
