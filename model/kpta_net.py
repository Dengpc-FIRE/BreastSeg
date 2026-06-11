from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kpta_blocks import ConvBlock, KineticEncoder, UpBlock
from .kpta_utils import EnhancementMapBuilder, PhaseIndices


class PseudoKineticMapBuilder(EnhancementMapBuilder):
    """Build pseudo-kinetic maps for pixel-wise temporal attention."""

    @property
    def expected_channels(self) -> int:
        if self.disable_kinetic_maps:
            return 1
        t = self.phase_indices.num_dynamic_phases
        return 2 * t + 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if self.disable_kinetic_maps:
            return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

        phases = self.split_phases(x)
        pre = phases["pre"]
        post = phases["post"]
        enhancement = (post - pre) / (pre.abs() + self.eps)
        if self.clip_value is not None and self.clip_value > 0:
            enhancement = enhancement.clamp(-self.clip_value, self.clip_value)

        peak, peak_idx = enhancement.max(dim=1, keepdim=True)
        mean = enhancement.mean(dim=1, keepdim=True)
        wash_in = enhancement[:, :1]
        if enhancement.shape[1] > 1:
            ttp = peak_idx.to(enhancement.dtype) / float(enhancement.shape[1] - 1)
            wash_out = enhancement[:, -1:] - peak
        else:
            ttp = torch.ones_like(peak)
            wash_out = torch.zeros_like(peak)

        kernel = int(self.local_pool_kernel)
        if kernel % 2 == 0:
            kernel += 1
        local_mean = F.avg_pool2d(enhancement, kernel_size=kernel, stride=1, padding=kernel // 2)
        local_contrast = enhancement - local_mean

        maps = torch.cat([enhancement, peak, mean, wash_in, ttp, wash_out, local_contrast], dim=1)
        maps = torch.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0)
        return self._normalize(maps)


class PhaseSpecificEncoder(nn.Module):
    def __init__(self, num_phases: int, base_channels: int = 32, mode: str = "shared") -> None:
        super().__init__()
        self.num_phases = num_phases
        self.mode = mode
        self.encoder = _SinglePhaseEncoder(base_channels)
        self.phase_embedding = nn.Parameter(torch.zeros(num_phases, 1, 1, 1))
        if mode == "phase_projection":
            self.phase_projection = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=1) for _ in range(num_phases)])
        elif mode != "shared":
            raise ValueError(f"Unknown phase_encoder_mode: {mode}")
        else:
            self.phase_projection = None

    def forward(self, phase_stack: torch.Tensor) -> List[torch.Tensor]:
        b, t, h, w = phase_stack.shape
        if t > self.num_phases:
            raise ValueError(f"Got {t} phases, but encoder was initialized for {self.num_phases}.")
        phases = []
        for idx in range(t):
            phase = phase_stack[:, idx : idx + 1]
            if self.phase_projection is not None:
                phase = self.phase_projection[idx](phase)
            else:
                phase = phase + self.phase_embedding[idx].view(1, 1, 1, 1)
            phases.append(phase)
        merged = torch.cat(phases, dim=0)
        encoded = self.encoder(merged)
        return [feat.reshape(t, b, feat.shape[1], feat.shape[2], feat.shape[3]).permute(1, 0, 2, 3, 4) for feat in encoded]


class _SinglePhaseEncoder(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.enc0 = ConvBlock(1, channels[0])
        self.enc1 = ConvBlock(channels[0], channels[1])
        self.enc2 = ConvBlock(channels[1], channels[2])
        self.enc3 = ConvBlock(channels[2], channels[3])
        self.bottleneck = ConvBlock(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))
        return [x0, x1, x2, x3, x4]


class PixelWisePhaseAttention(nn.Module):
    def __init__(self, channels: int, num_phases: int, disabled: bool = False) -> None:
        super().__init__()
        self.disabled = disabled
        self.num_phases = num_phases
        self.score = nn.Sequential(
            nn.Conv2d(channels * num_phases + channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, num_phases, kernel_size=1),
        )

    def forward(self, phase_feats: torch.Tensor, kinetic_feat: torch.Tensor):
        # phase_feats: [B,T,C,H,W]
        b, t, c, h, w = phase_feats.shape
        if self.disabled or t == 1:
            attn = phase_feats.new_full((b, t, 1, h, w), 1.0 / float(t))
            return phase_feats.mean(dim=1), attn
        if t != self.num_phases:
            pad = phase_feats[:, -1:].repeat(1, self.num_phases - t, 1, 1, 1) if t < self.num_phases else phase_feats[:, : self.num_phases]
            phase_feats_for_score = torch.cat([phase_feats, pad], dim=1) if t < self.num_phases else pad
        else:
            phase_feats_for_score = phase_feats
        flat = phase_feats_for_score.reshape(b, self.num_phases * c, h, w)
        logits = self.score(torch.cat([flat, kinetic_feat], dim=1))[:, :t]
        attn = torch.softmax(logits, dim=1).unsqueeze(2)
        fused = (phase_feats * attn).sum(dim=1)
        return fused, attn


class UncertaintyGuidedBoundaryRefinement(nn.Module):
    def __init__(self, channels: int, disabled: bool = False) -> None:
        super().__init__()
        self.disabled = disabled
        self.boundary_feature = ConvBlock(channels, channels)
        self.boundary_gate = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, decoder_feature: torch.Tensor, uncertainty_logits: torch.Tensor):
        boundary_feature = self.boundary_feature(decoder_feature)
        if self.disabled:
            return decoder_feature, boundary_feature
        uncertainty_gate = torch.sigmoid(uncertainty_logits)
        refined = decoder_feature + uncertainty_gate * self.boundary_gate(boundary_feature) * boundary_feature
        return refined, boundary_feature


class KPTANet(nn.Module):
    """Kinetic-aware Pixel-wise Temporal Attention Network."""

    def __init__(
        self,
        in_channels: int = 17,
        base_channels: int = 32,
        phase_indices: Optional[Dict] = None,
        compute_subtraction_if_missing: bool = True,
        use_pseudo_kinetic_maps: bool = True,
        phase_encoder_mode: str = "shared",
        use_pixelwise_phase_attention: bool = True,
        use_uncertainty_head: bool = True,
        use_boundary_head: bool = True,
        use_uncertainty_refinement: bool = True,
        store_attention_maps: bool = True,
        return_dict: bool = True,
        kinetic_eps: float = 1e-6,
        local_pool_kernel: int = 7,
        kinetic_clip_value: Optional[float] = 5.0,
        kinetic_normalize: Optional[str] = "per_image",
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        ablation = ablation or {}
        self.in_channels = in_channels
        self.return_dict = return_dict
        self.store_attention_maps = store_attention_maps
        self.disable_kinetic_maps = bool(ablation.get("disable_pseudo_kinetic_maps", False)) or not use_pseudo_kinetic_maps
        self.disable_attention = bool(ablation.get("disable_pixelwise_phase_attention", False)) or not use_pixelwise_phase_attention
        self.disable_boundary = bool(ablation.get("disable_boundary_head", False)) or not use_boundary_head
        self.disable_uncertainty = bool(ablation.get("disable_uncertainty_head", False)) or not use_uncertainty_head
        self.disable_refinement = bool(ablation.get("disable_uncertainty_refinement", False)) or not use_uncertainty_refinement

        self.phase_indices = PhaseIndices.from_config(phase_indices)
        self.kinetic_builder = PseudoKineticMapBuilder(
            phase_indices=phase_indices,
            compute_subtraction_if_missing=compute_subtraction_if_missing,
            eps=kinetic_eps,
            local_pool_kernel=local_pool_kernel,
            clip_value=kinetic_clip_value,
            normalize=kinetic_normalize,
            disable_kinetic_maps=self.disable_kinetic_maps,
        )
        self.num_phases = 1 + self.phase_indices.num_dynamic_phases
        self.phase_encoder = PhaseSpecificEncoder(self.num_phases, base_channels, mode=phase_encoder_mode)
        self.kinetic_encoder = KineticEncoder(self.kinetic_builder.expected_channels, base_channels)

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.attention = nn.ModuleList(
            [PixelWisePhaseAttention(ch, self.num_phases, disabled=self.disable_attention) for ch in channels]
        )
        self.up3 = UpBlock(channels[4], channels[3], channels[3])
        self.up2 = UpBlock(channels[3], channels[2], channels[2])
        self.up1 = UpBlock(channels[2], channels[1], channels[1])
        self.up0 = UpBlock(channels[1], channels[0], channels[0])
        self.uncertainty_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.refinement = UncertaintyGuidedBoundaryRefinement(channels[0], disabled=self.disable_refinement)
        self.boundary_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.seg_head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_dict: Optional[bool] = None):
        if x.ndim != 4:
            raise ValueError(f"KPTANet expects [B,C,H,W], got {tuple(x.shape)}")
        return_dict = self.return_dict if return_dict is None else return_dict

        phases = self.kinetic_builder.split_phases(x)
        phase_stack = torch.cat([phases["pre"], phases["post"]], dim=1)
        kinetic_maps = self.kinetic_builder(x)
        phase_feats = self.phase_encoder(phase_stack)
        kinetic_feats = self.kinetic_encoder(kinetic_maps)

        fused = []
        attention_maps = []
        for level, attn in enumerate(self.attention):
            fused_feat, attention_map = attn(phase_feats[level], kinetic_feats[level])
            fused.append(fused_feat)
            if self.store_attention_maps:
                attention_maps.append(attention_map)

        x_dec = self.up3(fused[4], fused[3])
        x_dec = self.up2(x_dec, fused[2])
        x_dec = self.up1(x_dec, fused[1])
        x_dec = self.up0(x_dec, fused[0])

        uncertainty_logits = self.uncertainty_head(x_dec)
        if self.disable_uncertainty:
            uncertainty_logits = torch.zeros_like(uncertainty_logits)
        refined, boundary_feature = self.refinement(x_dec, uncertainty_logits)
        boundary_logits = self.boundary_head(boundary_feature)
        if self.disable_boundary:
            boundary_logits = torch.zeros_like(boundary_logits)
        seg_logits = self.seg_head(refined)

        if not return_dict:
            return seg_logits
        return {
            "seg_logits": seg_logits,
            "logits": seg_logits,
            "boundary_logits": boundary_logits,
            "uncertainty_logits": uncertainty_logits,
            "kinetic_maps": kinetic_maps,
            "attention_maps": attention_maps,
            "debug": {"num_phases": phase_stack.shape[1]},
        }
