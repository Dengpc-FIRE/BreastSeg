"""Shared utilities for KPTA models."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


@dataclass(frozen=True)
class PhaseIndices:
    pre: int = 0
    post: tuple = ()
    subtraction: tuple = ()

    @classmethod
    def from_config(cls, config: Optional[Dict]) -> "PhaseIndices":
        config = config or {}
        return cls(
            pre=int(config.get("pre", 0)),
            post=tuple(_as_list(config.get("post", []))),
            subtraction=tuple(_as_list(config.get("subtraction", []))),
        )

    @property
    def num_dynamic_phases(self) -> int:
        return max(len(self.post), len(self.subtraction), 1)


def safe_channel_select(x: torch.Tensor, indices: Iterable[int]) -> Optional[torch.Tensor]:
    valid = [idx for idx in indices if 0 <= idx < x.shape[1]]
    if not valid:
        return None
    return x[:, valid, :, :]


def boundary_target_2d(mask: torch.Tensor, thickness: int = 3) -> torch.Tensor:
    """2D morphological gradient using max_pool2d."""
    if thickness <= 1:
        kernel_size = 3
    else:
        kernel_size = int(thickness)
        if kernel_size % 2 == 0:
            kernel_size += 1
    pad = kernel_size // 2
    mask = mask.float().clamp(0, 1)
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


class EnhancementMapBuilder(nn.Module):
    """Build 2D pseudo-kinetic maps from pre/post/subtraction DCE channels."""

    def __init__(
        self,
        phase_indices: Optional[Dict] = None,
        compute_subtraction_if_missing: bool = True,
        eps: float = 1e-6,
        local_pool_kernel: int = 7,
        clip_value: Optional[float] = 5.0,
        normalize: Optional[str] = "per_image",
        include_mean_peak: bool = True,
        disable_kinetic_maps: bool = False,
    ) -> None:
        super().__init__()
        self.phase_indices = PhaseIndices.from_config(phase_indices)
        self.compute_subtraction_if_missing = compute_subtraction_if_missing
        self.eps = eps
        self.local_pool_kernel = local_pool_kernel
        self.clip_value = clip_value
        self.normalize = normalize
        self.include_mean_peak = include_mean_peak
        self.disable_kinetic_maps = disable_kinetic_maps

    @property
    def expected_channels(self) -> int:
        if self.disable_kinetic_maps:
            return 1
        t = self.phase_indices.num_dynamic_phases
        return 3 * t + (2 if self.include_mean_peak else 0)

    def split_phases(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pre_idx = self.phase_indices.pre
        if pre_idx < 0 or pre_idx >= x.shape[1]:
            pre_idx = 0
        pre = x[:, pre_idx : pre_idx + 1, :, :]

        post = safe_channel_select(x, self.phase_indices.post)
        sub = safe_channel_select(x, self.phase_indices.subtraction)

        if post is None and sub is not None:
            post = pre + sub
        if sub is None and post is not None and self.compute_subtraction_if_missing:
            sub = post - pre
        if post is None and sub is None:
            post = pre
            sub = torch.zeros_like(pre)

        assert post is not None and sub is not None
        t = max(post.shape[1], sub.shape[1])
        if post.shape[1] != t:
            post = self._match_phase_count(post, t)
        if sub.shape[1] != t:
            sub = self._match_phase_count(sub, t)
        return {"pre": pre, "post": post, "subtraction": sub}

    @staticmethod
    def _match_phase_count(x: torch.Tensor, count: int) -> torch.Tensor:
        if x.shape[1] == count:
            return x
        if x.shape[1] == 1:
            return x.repeat(1, count, 1, 1)
        if x.shape[1] > count:
            return x[:, :count]
        pad = x[:, -1:].repeat(1, count - x.shape[1], 1, 1)
        return torch.cat([x, pad], dim=1)

    def _normalize(self, maps: torch.Tensor) -> torch.Tensor:
        if self.normalize in (None, "none", False):
            return maps
        if self.normalize == "per_batch":
            dims = (0, 2, 3)
        else:
            dims = (2, 3)
        mean = maps.mean(dim=dims, keepdim=True)
        std = maps.std(dim=dims, keepdim=True, unbiased=False)
        return (maps - mean) / (std + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if self.disable_kinetic_maps:
            return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

        phases = self.split_phases(x)
        pre = phases["pre"]
        post = phases["post"]
        sub = phases["subtraction"]

        computed_sub = post - pre
        subtraction = sub if sub is not None else computed_sub
        enhancement = computed_sub / (pre.abs() + self.eps)
        if self.clip_value is not None and self.clip_value > 0:
            enhancement = enhancement.clamp(-self.clip_value, self.clip_value)
            subtraction = subtraction.clamp(-self.clip_value, self.clip_value)

        kernel = int(self.local_pool_kernel)
        if kernel % 2 == 0:
            kernel += 1
        local_mean = F.avg_pool2d(enhancement, kernel_size=kernel, stride=1, padding=kernel // 2)
        local_contrast = enhancement - local_mean

        maps = [subtraction, enhancement, local_contrast]
        if self.include_mean_peak:
            maps.extend([enhancement.mean(dim=1, keepdim=True), enhancement.amax(dim=1, keepdim=True)])

        kinetic_maps = torch.cat(maps, dim=1)
        kinetic_maps = torch.nan_to_num(kinetic_maps, nan=0.0, posinf=0.0, neginf=0.0)
        return self._normalize(kinetic_maps)
