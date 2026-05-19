from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dce_kinetic_utils import EnhancementMapBuilder, PhaseIndices, safe_channel_select
from .sg_ktfnet import ConvBlock, UpBlock


class PhaseDropout(nn.Module):
    """Randomly remove dynamic DCE phases during training."""

    def __init__(
        self,
        phase_indices: Optional[Dict] = None,
        enabled: bool = True,
        drop_prob: float = 0.3,
        replacement: str = "zero",
        min_available_post_phases: int = 1,
        disabled: bool = False,
    ) -> None:
        super().__init__()
        self.phase_indices = PhaseIndices.from_config(phase_indices)
        self.enabled = enabled
        self.drop_prob = drop_prob
        self.replacement = replacement
        self.min_available_post_phases = min_available_post_phases
        self.disabled = disabled

    @property
    def num_dynamic_phases(self) -> int:
        return self.phase_indices.num_dynamic_phases

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        t = self.num_dynamic_phases
        phase_mask = x.new_ones((b, t))
        if not self.training or self.disabled or not self.enabled or self.drop_prob <= 0 or t <= 1:
            return x, phase_mask

        dropped = torch.rand((b, t), device=x.device) < self.drop_prob
        keep_count = (~dropped).sum(dim=1)
        min_keep = min(max(1, self.min_available_post_phases), t)
        for sample_idx in range(b):
            if keep_count[sample_idx] < min_keep:
                order = torch.randperm(t, device=x.device)
                dropped[sample_idx, order[:min_keep]] = False
        phase_mask = (~dropped).to(x.dtype)

        out = x.clone()
        for sample_idx in range(b):
            for phase_idx in range(t):
                if phase_mask[sample_idx, phase_idx] > 0:
                    continue
                for channel_idx in self._phase_channels(phase_idx):
                    if 0 <= channel_idx < out.shape[1] and channel_idx != self.phase_indices.pre:
                        out[sample_idx : sample_idx + 1, channel_idx : channel_idx + 1] = self._replacement(
                            out[sample_idx : sample_idx + 1], channel_idx
                        )
        return out, phase_mask

    def _phase_channels(self, phase_idx: int) -> List[int]:
        channels = []
        if self.phase_indices.post:
            channels.append(self.phase_indices.post[min(phase_idx, len(self.phase_indices.post) - 1)])
        if self.phase_indices.subtraction:
            channels.append(self.phase_indices.subtraction[min(phase_idx, len(self.phase_indices.subtraction) - 1)])
        return list(dict.fromkeys(channels))

    def _replacement(self, sample: torch.Tensor, channel_idx: int) -> torch.Tensor:
        if self.replacement == "pre":
            pre_idx = self.phase_indices.pre if 0 <= self.phase_indices.pre < sample.shape[1] else 0
            return sample[:, pre_idx : pre_idx + 1]
        if self.replacement == "channel_mean":
            return sample[:, channel_idx : channel_idx + 1].mean(dim=(2, 3), keepdim=True).expand_as(
                sample[:, channel_idx : channel_idx + 1]
            )
        return torch.zeros_like(sample[:, channel_idx : channel_idx + 1])


class KineticPriorEncoder(nn.Module):
    """Encode available enhancement dynamics into multi-scale features and a global code."""

    def __init__(
        self,
        phase_indices: Optional[Dict] = None,
        base_channels: int = 32,
        eps: float = 1e-6,
        use_phase_embedding: bool = True,
        max_phases: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.phase_indices = PhaseIndices.from_config(phase_indices)
        self.eps = eps
        self.use_phase_embedding = use_phase_embedding
        self.num_phases = max_phases or self.phase_indices.num_dynamic_phases
        self.phase_embedding = nn.Parameter(torch.zeros(self.num_phases, 1, 1, 1))
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.enc0 = ConvBlock(2, channels[0])
        self.enc1 = ConvBlock(channels[0], channels[1])
        self.enc2 = ConvBlock(channels[1], channels[2])
        self.enc3 = ConvBlock(channels[2], channels[3])
        self.bottleneck = ConvBlock(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)
        self.global_proj = nn.Sequential(
            nn.Linear(channels[4], channels[4]),
            nn.GELU(),
            nn.Linear(channels[4], channels[4]),
        )

    def forward(self, x: torch.Tensor, phase_mask: Optional[torch.Tensor] = None):
        pre = self._pre(x)
        post = safe_channel_select(x, self.phase_indices.post)
        sub = safe_channel_select(x, self.phase_indices.subtraction)
        if post is None and sub is not None:
            post = pre + sub
        if sub is None and post is not None:
            sub = post - pre
        if post is None and sub is None:
            post = pre
            sub = torch.zeros_like(pre)
        assert post is not None and sub is not None

        t = max(post.shape[1], sub.shape[1], 1)
        post = self._match_phase_count(post, t)
        sub = self._match_phase_count(sub, t)
        phase_mask = self._prepare_mask(phase_mask, x.shape[0], t, x.device, x.dtype)

        enhancement = (post - pre) / (pre.abs() + self.eps)
        enhancement = torch.nan_to_num(enhancement, nan=0.0, posinf=0.0, neginf=0.0)
        phase_inputs = []
        for idx in range(t):
            e = enhancement[:, idx : idx + 1]
            s = sub[:, idx : idx + 1]
            if self.use_phase_embedding and idx < self.phase_embedding.shape[0]:
                e = e + self.phase_embedding[idx].view(1, 1, 1, 1)
            phase_inputs.append(torch.cat([e, s], dim=1))

        merged = torch.cat(phase_inputs, dim=0)
        x0 = self.enc0(merged)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))
        stacked = [self._unstack(feat, x.shape[0], t) for feat in [x0, x1, x2, x3, x4]]
        features = [self._masked_mean(feat, phase_mask) for feat in stacked]
        global_code = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        global_code = self.global_proj(global_code)
        return features, global_code, enhancement, stacked[0]

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        pre_idx = self.phase_indices.pre
        if pre_idx < 0 or pre_idx >= x.shape[1]:
            pre_idx = 0
        return x[:, pre_idx : pre_idx + 1]

    @staticmethod
    def _match_phase_count(x: torch.Tensor, count: int) -> torch.Tensor:
        if x.shape[1] == count:
            return x
        if x.shape[1] == 1:
            return x.repeat(1, count, 1, 1)
        if x.shape[1] > count:
            return x[:, :count]
        return torch.cat([x, x[:, -1:].repeat(1, count - x.shape[1], 1, 1)], dim=1)

    @staticmethod
    def _prepare_mask(mask, batch: int, count: int, device, dtype) -> torch.Tensor:
        if mask is None:
            return torch.ones((batch, count), device=device, dtype=dtype)
        if mask.shape[1] == count:
            return mask.to(device=device, dtype=dtype)
        if mask.shape[1] > count:
            return mask[:, :count].to(device=device, dtype=dtype)
        pad = mask[:, -1:].repeat(1, count - mask.shape[1])
        return torch.cat([mask, pad], dim=1).to(device=device, dtype=dtype)

    @staticmethod
    def _unstack(feat: torch.Tensor, batch: int, phases: int) -> torch.Tensor:
        return feat.reshape(phases, batch, feat.shape[1], feat.shape[2], feat.shape[3]).permute(1, 0, 2, 3, 4)

    @staticmethod
    def _masked_mean(feat: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        weight = phase_mask[:, :, None, None, None]
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (feat * weight).sum(dim=1) / denom


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.enc0 = ConvBlock(in_channels, channels[0])
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


class KineticPriorFusion(nn.Module):
    def __init__(self, channels: int, mode: str = "gated", disabled: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.disabled = disabled
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, image_feat: torch.Tensor, kinetic_feat: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return image_feat
        if kinetic_feat.shape[-2:] != image_feat.shape[-2:]:
            kinetic_feat = F.interpolate(kinetic_feat, size=image_feat.shape[-2:], mode="bilinear", align_corners=False)
        if self.mode == "cross_attention":
            return image_feat + self._cross_attention(image_feat, kinetic_feat)
        gate = self.gate(torch.cat([image_feat, kinetic_feat], dim=1))
        return image_feat + gate * kinetic_feat

    def _cross_attention(self, image_feat: torch.Tensor, kinetic_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image_feat.shape
        if kinetic_feat.shape[-1] * kinetic_feat.shape[-2] > 256:
            kinetic_feat = F.adaptive_avg_pool2d(kinetic_feat, output_size=(16, 16))
        q = self.q(image_feat).flatten(2).transpose(1, 2)
        k = self.k(kinetic_feat).flatten(2)
        v = self.v(kinetic_feat).flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) / (c ** 0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return out


class TemporalContrastiveHead(nn.Module):
    def __init__(self, temperature: float = 0.1, min_tumor_pixels: int = 20) -> None:
        super().__init__()
        self.temperature = temperature
        self.min_tumor_pixels = min_tumor_pixels

    def forward(self, phase_features: torch.Tensor, mask: Optional[torch.Tensor], phase_mask: Optional[torch.Tensor] = None):
        # phase_features: [B,T,C,H,W] from enhancement-conditioned phase maps.
        return {
            "phase_features": phase_features,
            "mask": mask,
            "phase_mask": phase_mask,
            "temperature": self.temperature,
            "min_tumor_pixels": self.min_tumor_pixels,
        }


class KPRNet(nn.Module):
    """Kinetic Prior-guided Robust Segmentation Network."""

    def __init__(
        self,
        in_channels: int = 17,
        base_channels: int = 32,
        phase_indices: Optional[Dict] = None,
        compute_subtraction_if_missing: bool = True,
        use_kinetic_prior_encoder: bool = True,
        fusion_mode: str = "gated",
        use_phase_embedding: bool = True,
        support_missing_phase: bool = True,
        return_dict: bool = True,
        phase_dropout: Optional[Dict] = None,
        kinetic_eps: float = 1e-6,
        contrastive_temperature: float = 0.1,
        min_tumor_pixels: int = 20,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        ablation = ablation or {}
        self.in_channels = in_channels
        self.return_dict = return_dict
        self.phase_indices = PhaseIndices.from_config(phase_indices)
        self.support_missing_phase = support_missing_phase
        self.disable_phase_dropout = bool(ablation.get("disable_phase_dropout", False))
        self.disable_kinetic_prior_encoder = (
            bool(ablation.get("disable_kinetic_prior_encoder", False)) or not use_kinetic_prior_encoder
        )
        self.disable_kinetic_fusion = bool(ablation.get("disable_kinetic_fusion", False))
        phase_dropout = phase_dropout or {}
        self.phase_dropout = PhaseDropout(
            phase_indices=phase_indices,
            disabled=self.disable_phase_dropout,
            **phase_dropout,
        )
        self.kinetic_encoder = KineticPriorEncoder(
            phase_indices=phase_indices,
            base_channels=base_channels,
            eps=kinetic_eps,
            use_phase_embedding=use_phase_embedding,
        )
        self.image_encoder = ImageEncoder(in_channels, base_channels)
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.fusions = nn.ModuleList(
            [KineticPriorFusion(ch, mode=fusion_mode, disabled=self.disable_kinetic_fusion) for ch in channels]
        )
        self.up3 = UpBlock(channels[4], channels[3], channels[3])
        self.up2 = UpBlock(channels[3], channels[2], channels[2])
        self.up1 = UpBlock(channels[2], channels[1], channels[1])
        self.up0 = UpBlock(channels[1], channels[0], channels[0])
        self.seg_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.contrastive_head = TemporalContrastiveHead(
            temperature=contrastive_temperature,
            min_tumor_pixels=min_tumor_pixels,
        )
        self.kinetic_map_builder = EnhancementMapBuilder(
            phase_indices=phase_indices,
            compute_subtraction_if_missing=compute_subtraction_if_missing,
            eps=kinetic_eps,
            include_mean_peak=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        phase_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        if x.ndim != 4:
            raise ValueError(f"KPRNet expects [B,C,H,W], got {tuple(x.shape)}")
        return_dict = self.return_dict if return_dict is None else return_dict

        if phase_mask is None:
            x_used, phase_mask = self.phase_dropout(x)
        else:
            x_used = self._apply_phase_mask(x, phase_mask)

        kinetic_features, global_code, enhancement, phase_contrast_features = self.kinetic_encoder(x_used, phase_mask)
        if self.disable_kinetic_prior_encoder:
            kinetic_features = [torch.zeros_like(feat) for feat in kinetic_features]
            global_code = torch.zeros_like(global_code)

        image_features = self.image_encoder(x_used)
        fused = [fusion(img, kin) for fusion, img, kin in zip(self.fusions, image_features, kinetic_features)]

        x_dec = self.up3(fused[4], fused[3])
        x_dec = self.up2(x_dec, fused[2])
        x_dec = self.up1(x_dec, fused[1])
        x_dec = self.up0(x_dec, fused[0])
        seg_logits = self.seg_head(x_dec)

        contrastive_embeddings = None
        if self.training or mask is not None:
            contrastive_embeddings = self.contrastive_head(
                phase_contrast_features,
                mask,
                phase_mask,
            )

        if not return_dict:
            return seg_logits
        return {
            "seg_logits": seg_logits,
            "logits": seg_logits,
            "kinetic_features": kinetic_features,
            "global_kinetic_code": global_code,
            "contrastive_embeddings": contrastive_embeddings,
            "phase_mask": phase_mask,
            "kinetic_maps": self.kinetic_map_builder(x_used),
            "debug": {"num_dynamic_phases": phase_mask.shape[1]},
        }

    def _apply_phase_mask(self, x: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        t = phase_mask.shape[1]
        for sample_idx in range(x.shape[0]):
            for phase_idx in range(t):
                if phase_mask[sample_idx, phase_idx] > 0:
                    continue
                for channel_idx in self.phase_dropout._phase_channels(phase_idx):
                    if 0 <= channel_idx < out.shape[1] and channel_idx != self.phase_indices.pre:
                        out[sample_idx : sample_idx + 1, channel_idx : channel_idx + 1] = 0
        return out

class SwinHR(KPRNet):
    """Compatibility wrapper for train_swinhr.py --model_name kpr_net."""

    def __init__(self, in_channels: int = 1, attn_channels: int = 8, *args, **kwargs) -> None:
        total_channels = int(kwargs.pop("total_channels", in_channels + attn_channels))
        if total_channels >= 17:
            phase_indices = {"pre": 0, "post": list(range(1, 9)), "subtraction": list(range(9, 17))}
        else:
            phase_indices = {"pre": 0, "post": [], "subtraction": list(range(1, total_channels))}
        super().__init__(in_channels=total_channels, phase_indices=phase_indices, *args, **kwargs)
