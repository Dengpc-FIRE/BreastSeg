"""Inference-only whole-breast region constraint.

This module deliberately stays outside the tumor model and training loss.  It
reconstructs a pre-contrast volume for each case, runs a frozen nnU-Net v1
whole-breast model once, caches the result, and returns the mask corresponding
to each center slice.

The tumor training path is not changed:

    tumor model input -> tumor logits -> tumor loss

Only validation/test probabilities are constrained:

    sigmoid(tumor_logits) * whole_breast_mask
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SLICE_PATTERN = re.compile(r"^(?P<case>.+)_(?P<slice>p-\d+|\d+)$", re.IGNORECASE)


def _resolve_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _natural_key(value: str) -> Tuple:
    """Sort p-2 before p-10 while preserving non-numeric text."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def split_case_and_slice(file_name: str) -> Tuple[str, str]:
    """Split ``BreaDM-Ma-1809_p-038.npy`` into case and slice identifiers."""
    stem = Path(file_name).stem
    match = SLICE_PATTERN.match(stem)
    if match:
        return match.group("case"), match.group("slice")
    if "_" in stem:
        return tuple(stem.rsplit("_", 1))  # type: ignore[return-value]
    return stem, stem


class WholeBreastConstraint:
    """Generate and cache whole-breast masks for validation/test inference.

    Loading of nnU-Net is lazy. Constructing this object before training does
    not load the checkpoint; the model is initialized only when validation or
    testing first requests a breast mask.
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device,
        output_path: Optional[Path] = None,
    ) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.backend = str(
            self.config.get("backend", "nnunet_v1")
        ).lower()
        self.device = device
        self.pre_phase_index = int(self.config.get("pre_phase_index", 0))
        self.model_folder = _resolve_path(
            self.config.get(
                "model_folder",
                "./nnUNet/3d_fullres/Task555_breast/"
                "nnUNetTrainerV2__nnUNetPlansv2.1",
            )
        )
        self.raw_data_root = _resolve_path(
            self.config.get("raw_data_root", "./seg")
        )
        self.raw_pre_dir_name = str(
            self.config.get("raw_pre_dir_name", "VIBRANT")
        )
        self.folds = self.config.get("folds", [0])
        self.checkpoint_name = str(
            self.config.get("checkpoint_name", "model_best")
        )
        self.foreground_class = int(self.config.get("foreground_class", 1))
        self.breast_threshold = float(
            self.config.get("breast_threshold", 0.5)
        )
        self.tumor_threshold = float(
            self.config.get("tumor_threshold", 0.5)
        )
        self.step_size = float(self.config.get("step_size", 0.5))
        self.use_gaussian = bool(self.config.get("use_gaussian", True))
        self.do_mirroring = bool(self.config.get("do_mirroring", False))
        self.mixed_precision = bool(
            self.config.get("mixed_precision", device.type == "cuda")
        )
        self.normalization = str(
            self.config.get("normalization", "nnunet_nonct")
        ).lower()
        self.normalize_nonzero_only = bool(
            self.config.get("normalize_nonzero_only", False)
        )
        self.clip_percentiles = self.config.get(
            "clip_percentiles", [0.5, 99.5]
        )
        self.transpose_axes = tuple(
            int(axis) for axis in self.config.get("transpose_axes", [0, 1, 2])
        )
        self.flip_axes = tuple(
            int(axis) for axis in self.config.get("flip_axes", [])
        )
        self.dilation_pixels = int(
            self.config.get("dilation_pixels", 5)
        )
        self.dilation_slices = int(
            self.config.get("dilation_slices", 0)
        )
        self.min_breast_fraction = float(
            self.config.get("min_breast_fraction", 0.01)
        )
        self.max_breast_fraction = float(
            self.config.get("max_breast_fraction", 0.95)
        )
        self.invalid_mask_policy = str(
            self.config.get("invalid_mask_policy", "full_mask")
        ).lower()
        self.runtime_failure_policy = str(
            self.config.get("runtime_failure_policy", "error")
        ).lower()
        self.verbose = bool(self.config.get("verbose", True))
        cache_cfg = self.config.get("cache", {})
        self.use_disk_cache = bool(cache_cfg.get("enabled", True))
        default_cache = (
            Path(output_path) / "whole_breast_cache"
            if output_path is not None
            else PROJECT_ROOT / "whole_breast_cache"
        )
        self.cache_dir = _resolve_path(
            cache_cfg.get("dir", str(default_cache))
        )

        self._trainer = None
        self._slice_cache: Dict[str, np.ndarray] = {}
        self._case_files_cache: Dict[Tuple[str, str], List[str]] = {}
        self._warned_messages = set()

        if not 0.0 < self.breast_threshold < 1.0:
            raise ValueError("whole_breast.breast_threshold must be in (0,1)")
        if not 0.0 < self.tumor_threshold < 1.0:
            raise ValueError("whole_breast.tumor_threshold must be in (0,1)")
        if sorted(self.transpose_axes) != [0, 1, 2]:
            raise ValueError(
                "whole_breast.transpose_axes must be a permutation of [0,1,2]"
            )
        if any(axis not in {0, 1, 2} for axis in self.flip_axes):
            raise ValueError("whole_breast.flip_axes may contain only 0, 1, 2")
        if self.backend != "nnunet_v1":
            raise ValueError(
                "Only whole_breast.backend='nnunet_v1' is currently supported"
            )
        if self.invalid_mask_policy not in {"full_mask", "error"}:
            raise ValueError(
                "whole_breast.invalid_mask_policy must be full_mask or error"
            )
        if self.runtime_failure_policy not in {"full_mask", "error"}:
            raise ValueError(
                "whole_breast.runtime_failure_policy must be full_mask or error"
            )

    def constrain_probabilities(
        self,
        tumor_probabilities: torch.Tensor,
        names: Sequence[str],
        dataset,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multiply tumor probabilities by center-slice breast masks."""
        if not self.enabled:
            ones = torch.ones_like(tumor_probabilities)
            return tumor_probabilities, ones
        breast_masks = self.get_center_masks(
            names=names,
            dataset=dataset,
            spatial_shape=tumor_probabilities.shape[-2:],
            dtype=tumor_probabilities.dtype,
            device=tumor_probabilities.device,
        )
        return tumor_probabilities * breast_masks, breast_masks

    def get_center_masks(
        self,
        names: Sequence[str],
        dataset,
        spatial_shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return masks with shape [B,1,H,W] in the tumor tensor's space."""
        masks = []
        for name in names:
            key = self._slice_key(dataset, name)
            if key not in self._slice_cache:
                self._predict_and_cache_case(dataset, name, spatial_shape)
            mask = self._slice_cache.get(key)
            if mask is None:
                mask = np.ones(tuple(spatial_shape), dtype=np.float32)
            if mask.shape != tuple(spatial_shape):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (int(spatial_shape[1]), int(spatial_shape[0])),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks.append(mask.astype(np.float32, copy=False))
        batch = np.stack(masks, axis=0)[:, None]
        return torch.from_numpy(batch).to(device=device, dtype=dtype)

    def _predict_and_cache_case(
        self,
        dataset,
        requested_name: str,
        spatial_shape: Sequence[int],
    ) -> None:
        case_id, _ = split_case_and_slice(requested_name)
        split_name = self._dataset_split_name(dataset)
        case_files = self._case_files(dataset, case_id)
        if not case_files:
            case_files = [requested_name]

        cached = self._load_disk_cache(split_name, case_id)
        if cached is not None:
            cached_names, cached_masks = cached
            for name, mask in zip(cached_names, cached_masks):
                self._slice_cache[self._slice_key(dataset, name)] = mask
            if self._slice_key(dataset, requested_name) in self._slice_cache:
                return

        try:
            volume, volume_names = self._load_precontrast_volume(
                dataset=dataset,
                split_name=split_name,
                case_id=case_id,
                case_files=case_files,
                spatial_shape=spatial_shape,
            )
            breast_volume = self._predict_volume(volume)
            breast_volume = self._postprocess_and_validate(
                breast_volume,
                case_id,
            )
        except Exception as exc:
            if self.runtime_failure_policy == "error":
                raise RuntimeError(
                    f"Whole-breast inference failed for case {case_id}: {exc}"
                ) from exc
            self._warn_once(
                f"Whole-breast inference failed for {case_id}; using an "
                f"unconstrained full-image mask. Reason: {exc}"
            )
            volume_names = case_files
            breast_volume = np.ones(
                (len(volume_names), int(spatial_shape[0]), int(spatial_shape[1])),
                dtype=np.uint8,
            )

        cached_names: List[str] = []
        cached_masks: List[np.ndarray] = []
        for name, mask in zip(volume_names, breast_volume):
            output_name = name if name.endswith(".npy") else f"{name}.npy"
            if output_name not in case_files:
                continue
            mask = mask.astype(np.float32, copy=False)
            if not np.any(mask):
                if self.invalid_mask_policy == "error":
                    raise ValueError(
                        f"Empty whole-breast center mask for {output_name}"
                    )
                self._warn_once(
                    f"Empty whole-breast mask for {output_name}; using a "
                    "full-image mask to avoid deleting a true tumor."
                )
                mask = np.ones_like(mask, dtype=np.float32)
            self._slice_cache[self._slice_key(dataset, output_name)] = mask
            cached_names.append(output_name)
            cached_masks.append(mask)

        requested_key = self._slice_key(dataset, requested_name)
        if requested_key not in self._slice_cache:
            self._warn_once(
                f"Could not map {requested_name} back to the whole-breast "
                "volume; using a full-image mask."
            )
            self._slice_cache[requested_key] = np.ones(
                tuple(spatial_shape), dtype=np.float32
            )
        if cached_masks:
            self._save_disk_cache(
                split_name,
                case_id,
                cached_names,
                np.stack(cached_masks),
            )

    def _load_precontrast_volume(
        self,
        dataset,
        split_name: str,
        case_id: str,
        case_files: Sequence[str],
        spatial_shape: Sequence[int],
    ) -> Tuple[np.ndarray, List[str]]:
        """Prefer the complete raw VIBRANT volume, then fall back to .npy."""
        raw_dir = None
        if self.raw_data_root is not None:
            raw_dir = (
                self.raw_data_root
                / split_name
                / "images"
                / case_id
                / self.raw_pre_dir_name
            )
        if raw_dir is not None and raw_dir.is_dir():
            raw_files = sorted(
                (
                    path
                    for path in raw_dir.iterdir()
                    if path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda path: _natural_key(path.stem),
            )
            if raw_files:
                slices = []
                names = []
                target_h, target_w = int(spatial_shape[0]), int(spatial_shape[1])
                for path in raw_files:
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    if image.shape != (target_h, target_w):
                        image = cv2.resize(
                            image,
                            (target_w, target_h),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    slices.append(image.astype(np.float32))
                    names.append(f"{case_id}_{path.stem}.npy")
                if slices:
                    return np.stack(slices), names

        data_dir = Path(dataset.data_dir)
        slices = []
        names = []
        for name in case_files:
            array = np.load(data_dir / name)
            slices.append(self._extract_center_pre(array, name))
            names.append(name)
        if not slices:
            raise ValueError(f"No pre-contrast slices found for {case_id}")
        return np.stack(slices), names

    def _extract_center_pre(
        self,
        array: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if array.ndim == 4:
            center = array.shape[0] // 2
            if self.pre_phase_index >= array.shape[1]:
                raise IndexError(
                    f"pre_phase_index={self.pre_phase_index} is outside "
                    f"2.5D sample {name} with T={array.shape[1]}"
                )
            return array[center, self.pre_phase_index].astype(np.float32)
        if array.ndim == 3:
            if self.pre_phase_index >= array.shape[-1]:
                raise IndexError(
                    f"pre_phase_index={self.pre_phase_index} is outside "
                    f"2D sample {name} with C={array.shape[-1]}"
                )
            return array[..., self.pre_phase_index].astype(np.float32)
        raise ValueError(
            f"Unsupported sample shape {array.shape} for whole-breast input"
        )

    def _predict_volume(self, volume: np.ndarray) -> np.ndarray:
        """Run nnU-Net on one [Z,H,W] pre-contrast volume."""
        original_shape = volume.shape
        volume = self._prepare_volume(volume)
        self._ensure_nnunet_loaded()
        data = volume[None].astype(np.float32, copy=False)
        predict_fn = (
            self._trainer.predict_preprocessed_data_return_seg_and_softmax
        )
        predict_kwargs = {
            "do_mirroring": self.do_mirroring,
            "mirror_axes": (
                self._trainer.data_aug_params.get("mirror_axes")
                if self.do_mirroring
                else None
            ),
            "use_sliding_window": True,
            "step_size": self.step_size,
            "use_gaussian": self.use_gaussian,
            "all_in_gpu": False,
            "mixed_precision": self.mixed_precision,
            "verbose": self.verbose,
        }
        # nnU-Net v1 minor releases differ slightly in prediction arguments.
        # Pass only parameters supported by the installed release.
        signature = inspect.signature(predict_fn)
        if not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            predict_kwargs = {
                key: value
                for key, value in predict_kwargs.items()
                if key in signature.parameters
            }
        segmentation, softmax = predict_fn(data, **predict_kwargs)
        if (
            isinstance(softmax, np.ndarray)
            and softmax.ndim == 4
            and self.foreground_class < softmax.shape[0]
        ):
            mask = softmax[self.foreground_class] >= self.breast_threshold
        else:
            mask = np.asarray(segmentation) == self.foreground_class
        mask = self._restore_orientation(mask.astype(np.uint8))
        if mask.shape != original_shape:
            mask_tensor = torch.from_numpy(mask[None, None].astype(np.float32))
            mask = (
                F.interpolate(
                    mask_tensor,
                    size=original_shape,
                    mode="nearest",
                )[0, 0]
                .numpy()
                .astype(np.uint8)
            )
        return mask

    def _prepare_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = np.nan_to_num(volume.astype(np.float32))
        if self.clip_percentiles:
            low_q, high_q = (
                float(self.clip_percentiles[0]),
                float(self.clip_percentiles[1]),
            )
            valid = volume[np.isfinite(volume)]
            if valid.size:
                low, high = np.percentile(valid, [low_q, high_q])
                if high > low:
                    volume = np.clip(volume, low, high)
        if self.normalization in {"nnunet_nonct", "zscore", "z-score"}:
            selected = volume != 0 if self.normalize_nonzero_only else np.ones(
                volume.shape, dtype=bool
            )
            values = volume[selected]
            if values.size:
                mean = float(values.mean())
                std = float(values.std())
                volume[selected] = (volume[selected] - mean) / max(std, 1e-8)
                if self.normalize_nonzero_only:
                    volume[~selected] = 0.0
        elif self.normalization not in {"none", "identity"}:
            raise ValueError(
                f"Unsupported whole_breast.normalization: {self.normalization}"
            )
        volume = np.transpose(volume, self.transpose_axes)
        for axis in self.flip_axes:
            volume = np.flip(volume, axis=axis)
        return np.ascontiguousarray(volume)

    def _restore_orientation(self, mask: np.ndarray) -> np.ndarray:
        for axis in reversed(self.flip_axes):
            mask = np.flip(mask, axis=axis)
        inverse_axes = tuple(int(axis) for axis in np.argsort(self.transpose_axes))
        return np.ascontiguousarray(np.transpose(mask, inverse_axes))

    def _postprocess_and_validate(
        self,
        mask: np.ndarray,
        case_id: str,
    ) -> np.ndarray:
        mask_tensor = torch.from_numpy(mask[None, None].astype(np.float32))
        if self.dilation_pixels > 0 or self.dilation_slices > 0:
            kernel = (
                2 * self.dilation_slices + 1,
                2 * self.dilation_pixels + 1,
                2 * self.dilation_pixels + 1,
            )
            mask_tensor = F.max_pool3d(
                mask_tensor,
                kernel_size=kernel,
                stride=1,
                padding=(
                    self.dilation_slices,
                    self.dilation_pixels,
                    self.dilation_pixels,
                ),
            )
        mask = (mask_tensor[0, 0].numpy() > 0.5).astype(np.uint8)
        fraction = float(mask.mean())
        valid = (
            self.min_breast_fraction
            <= fraction
            <= self.max_breast_fraction
        )
        if not valid:
            message = (
                f"Whole-breast mask for {case_id} occupies {fraction:.2%} "
                f"of the volume, outside configured range "
                f"[{self.min_breast_fraction:.2%}, "
                f"{self.max_breast_fraction:.2%}]."
            )
            if self.invalid_mask_policy == "error":
                raise ValueError(message)
            self._warn_once(message + " Using a full-image mask.")
            mask = np.ones_like(mask, dtype=np.uint8)
        elif self.verbose:
            print(
                f"[whole_breast] case={case_id} "
                f"breast_fraction={fraction:.4f}"
            )
        return mask

    def _ensure_nnunet_loaded(self) -> None:
        if self._trainer is not None:
            return
        if self.device.type != "cuda":
            raise RuntimeError(
                "The bundled nnU-Net v1 3D predictor requires CUDA. "
                "Run evaluation with a CUDA device or precompute masks."
            )
        if self.model_folder is None or not self.model_folder.is_dir():
            raise FileNotFoundError(
                f"nnU-Net model folder not found: {self.model_folder}"
            )
        try:
            from nnunet.training.model_restore import (
                load_model_and_checkpoint_files,
            )
        except ImportError as exc:
            raise ImportError(
                "nnU-Net v1 is required for whole-breast inference. "
                "Install the optional dependencies with "
                "`pip install -r requirements-whole-breast.txt`."
            ) from exc

        trainer, checkpoint_params = load_model_and_checkpoint_files(
            str(self.model_folder),
            folds=self.folds,
            mixed_precision=self.mixed_precision,
            checkpoint_name=self.checkpoint_name,
        )
        if not checkpoint_params:
            raise RuntimeError(
                f"No checkpoint parameters loaded from {self.model_folder}"
            )
        trainer.load_checkpoint_ram(checkpoint_params[0], False)
        trainer.network.eval()
        for parameter in trainer.network.parameters():
            parameter.requires_grad_(False)
        self._trainer = trainer
        if self.verbose:
            print(
                "[whole_breast] Loaded frozen nnU-Net model: "
                f"{self.model_folder} ({self.checkpoint_name}, "
                f"folds={self.folds})"
            )

    def _case_files(self, dataset, case_id: str) -> List[str]:
        data_dir = str(Path(dataset.data_dir).resolve())
        cache_key = (data_dir, case_id)
        if cache_key in self._case_files_cache:
            return self._case_files_cache[cache_key]
        if hasattr(dataset, "ids"):
            names = [str(name) for name in dataset.ids]
        elif hasattr(dataset, "files"):
            names = [Path(path).name for path in dataset.files]
        else:
            names = [path.name for path in Path(dataset.data_dir).glob("*.npy")]
        case_names = [
            name for name in names if split_case_and_slice(name)[0] == case_id
        ]
        case_names.sort(key=lambda name: _natural_key(split_case_and_slice(name)[1]))
        self._case_files_cache[cache_key] = case_names
        return case_names

    @staticmethod
    def _dataset_split_name(dataset) -> str:
        data_dir = Path(dataset.data_dir)
        return data_dir.parent.name

    @staticmethod
    def _slice_key(dataset, name: str) -> str:
        return f"{Path(dataset.data_dir).resolve()}::{Path(name).name}"

    def _cache_path(self, split_name: str, case_id: str) -> Optional[Path]:
        if not self.use_disk_cache or self.cache_dir is None:
            return None
        settings = {
            "model_folder": str(self.model_folder),
            "checkpoint": self.checkpoint_name,
            "threshold": self.breast_threshold,
            "dilation_pixels": self.dilation_pixels,
            "dilation_slices": self.dilation_slices,
            "transpose_axes": self.transpose_axes,
            "flip_axes": self.flip_axes,
            "normalization": self.normalization,
            "normalize_nonzero_only": self.normalize_nonzero_only,
            "clip_percentiles": self.clip_percentiles,
        }
        digest = hashlib.sha1(
            json.dumps(settings, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
        return self.cache_dir / f"{split_name}__{safe_case}__{digest}.npz"

    def _load_disk_cache(
        self,
        split_name: str,
        case_id: str,
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        path = self._cache_path(split_name, case_id)
        if path is None or not path.is_file():
            return None
        try:
            cached = np.load(path, allow_pickle=False)
            names = [str(name) for name in cached["names"].tolist()]
            masks = cached["masks"].astype(np.float32)
            return names, masks
        except Exception as exc:
            self._warn_once(f"Ignoring invalid breast-mask cache {path}: {exc}")
            return None

    def _save_disk_cache(
        self,
        split_name: str,
        case_id: str,
        names: Sequence[str],
        masks: np.ndarray,
    ) -> None:
        path = self._cache_path(split_name, case_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            names=np.asarray(list(names)),
            masks=masks.astype(np.uint8),
        )

    def _warn_once(self, message: str) -> None:
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def build_whole_breast_constraint(
    config: Dict,
    device: torch.device,
    output_path: Optional[Path] = None,
) -> Optional[WholeBreastConstraint]:
    """Build a lazy inference constraint, or return None when disabled."""
    section = config.get("whole_breast", {})
    if not bool(section.get("enabled", False)):
        return None
    return WholeBreastConstraint(section, device=device, output_path=output_path)
