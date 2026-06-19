from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


COHORTS = ("DUKE", "ISPY1", "ISPY2", "NACT")


@dataclass(frozen=True)
class MamaMiaCase:
    case_id: str
    cohort: str
    image_dir: Path
    phase_paths: Tuple[Path, ...]
    mask_path: Path


def normalize_cohort(value: str) -> str:
    normalized = value.upper().replace("-", "").replace("_", "")
    if normalized == "ALL":
        return "ALL"
    if normalized == "DUCK":
        normalized = "DUKE"
    if normalized not in COHORTS:
        raise ValueError(f"Unknown cohort {value!r}; expected one of {COHORTS}")
    return normalized


def selected_cohorts(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return list(COHORTS)
    normalized = [normalize_cohort(value) for value in values]
    if "ALL" in normalized:
        return list(COHORTS)
    return [cohort for cohort in COHORTS if cohort in normalized]


def cohort_from_case_id(case_id: str) -> Optional[str]:
    lowered = case_id.lower()
    if lowered.startswith("duke_"):
        return "DUKE"
    if lowered.startswith("ispy1_"):
        return "ISPY1"
    if lowered.startswith("ispy2_"):
        return "ISPY2"
    if lowered.startswith("nact_"):
        return "NACT"
    return None


def _case_key(path: Path) -> str:
    return path.name.lower()


def _phase_index(path: Path) -> int:
    match = re.search(r"_(\d{4})\.nii(?:\.gz)?$", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 9999


def _find_mask(root: Path, case_id: str, mask_source: str) -> Optional[Path]:
    case_lower = case_id.lower()
    mask_dir = root / "segmentations" / mask_source
    for name in (f"{case_lower}.nii.gz", f"{case_id}.nii.gz"):
        path = mask_dir / name
        if path.exists():
            return path
    return None


def discover_cases(
    mama_root: str | Path,
    cohorts: Sequence[str],
    mask_source: str = "expert",
    require_mask: bool = True,
) -> List[MamaMiaCase]:
    root = Path(mama_root)
    image_root = root / "images"
    selected = set(selected_cohorts(cohorts))
    if not image_root.is_dir():
        raise FileNotFoundError(f"MAMA-MIA images directory not found: {image_root}")

    cases: List[MamaMiaCase] = []
    for image_dir in sorted([p for p in image_root.iterdir() if p.is_dir()], key=_case_key):
        cohort = cohort_from_case_id(image_dir.name)
        if cohort is None or cohort not in selected:
            continue
        phase_paths = tuple(sorted(image_dir.glob("*.nii.gz"), key=_phase_index))
        if len(phase_paths) < 2:
            continue
        mask_path = _find_mask(root, image_dir.name, mask_source)
        if mask_path is None:
            if require_mask:
                continue
            mask_path = Path()
        cases.append(
            MamaMiaCase(
                case_id=image_dir.name.lower(),
                cohort=cohort,
                image_dir=image_dir,
                phase_paths=phase_paths,
                mask_path=mask_path,
            )
        )
    return cases


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Please install nibabel: pip install nibabel") from exc
    data = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got {data.shape} for {path}")
    return np.transpose(data, (2, 0, 1))


def resize_volume(volume: np.ndarray, size: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    if volume.shape[-2:] == tuple(size):
        return volume.astype(np.float32, copy=False)
    resample = Image.NEAREST if nearest else Image.BILINEAR
    out = []
    for plane in volume:
        image = Image.fromarray(plane.astype(np.float32))
        image = image.resize((int(size[1]), int(size[0])), resample)
        out.append(np.asarray(image, dtype=np.float32))
    return np.stack(out, axis=0)


def _normalize_channels(volume: np.ndarray, mode: str) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return volume
    out = volume.copy()
    for channel in range(out.shape[0]):
        values = out[channel]
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            out[channel] = 0.0
            continue
        if mode == "minmax":
            lo, hi = float(valid.min()), float(valid.max())
            out[channel] = (values - lo) / (hi - lo + 1e-6)
        else:
            nonzero = valid[np.abs(valid) > 1e-6]
            stats = nonzero if nonzero.size else valid
            out[channel] = (values - float(stats.mean())) / (float(stats.std()) + 1e-6)
    return out


def resample_posts_to_8(posts: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Linearly resample arbitrary post-contrast phases to 8 BreastDM phases.

    BreastDM's 17-channel layout assumes 1 pre + 8 post + 8 subtraction maps.
    MAMA-MIA cases may have fewer or more post phases, so we normalize the time
    axis instead of padding with artificial zero phases or blindly truncating.
    If a case has exactly one post phase there is no temporal axis to
    interpolate, so that single post is repeated as a last-resort fallback.
    """
    if not posts:
        raise ValueError("At least one post-contrast phase is required.")
    if len(posts) == 8:
        return [post.astype(np.float32, copy=False) for post in posts]
    if len(posts) == 1:
        return [posts[0].astype(np.float32, copy=True) for _ in range(8)]

    timeline = np.linspace(0.0, float(len(posts) - 1), num=8)
    resampled: List[np.ndarray] = []
    for position in timeline:
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        if lower == upper:
            resampled.append(posts[lower].astype(np.float32, copy=True))
            continue
        weight = np.float32(position - lower)
        resampled.append(
            ((1.0 - weight) * posts[lower] + weight * posts[upper]).astype(np.float32)
        )
    return resampled


def load_case_17ch(
    case: MamaMiaCase,
    image_size: Tuple[int, int],
    normalize: str = "zscore",
) -> Tuple[np.ndarray, np.ndarray]:
    phase_volumes = [resize_volume(_load_nifti(path), image_size, nearest=False) for path in case.phase_paths]
    pre = phase_volumes[0]
    posts = resample_posts_to_8(phase_volumes[1:])
    subs = [post - pre for post in posts]
    image = np.stack([pre, *posts, *subs], axis=0).astype(np.float32)
    image = _normalize_channels(image, normalize)

    mask = resize_volume(_load_nifti(case.mask_path), image_size, nearest=True)
    mask = (mask > 0).astype(np.float32)
    return image, mask


def adapt_channel_count(image_17ch: np.ndarray, target_channels: int) -> np.ndarray:
    """Adapt canonical MAMA-MIA [17,D,H,W] to a model's configured channels.

    Canonical layout is 0=pre, 1..8=post, 9..16=subtraction. For legacy
    9-channel BreastDM models we use 0 + 9..16, i.e. pre plus subtraction maps.
    Other counts are handled by truncation/padding so experiments can run
    without modifying the dataset on disk.
    """
    if target_channels == image_17ch.shape[0]:
        return image_17ch
    if target_channels == 9 and image_17ch.shape[0] >= 17:
        return image_17ch[[0, *range(9, 17)]]
    if target_channels < image_17ch.shape[0]:
        return image_17ch[:target_channels]
    pad_count = target_channels - image_17ch.shape[0]
    pad = np.repeat(image_17ch[-1:], pad_count, axis=0)
    return np.concatenate([image_17ch, pad], axis=0)


def neighbor_indices(center: int, count: int, num_slices: int) -> List[int]:
    half = num_slices // 2
    return [min(max(center + offset, 0), count - 1) for offset in range(-half, half + 1)]
