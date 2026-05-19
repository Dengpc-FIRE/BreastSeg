from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dce_kinetic_utils import EnhancementMapBuilder, PhaseIndices


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedPhaseEncoder(nn.Module):
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


class KineticEncoder(nn.Module):
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


class SubtractionGuidedResidualFusion(nn.Module):
    """Gate post-pre residuals using subtraction and kinetic features."""

    def __init__(self, channels: int, aggregation: str = "attention", disabled: bool = False) -> None:
        super().__init__()
        self.aggregation = aggregation
        self.disabled = disabled
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.attention_score = nn.Conv2d(channels * 2, 1, kernel_size=1)

    def forward(
        self,
        pre_feat: torch.Tensor,
        post_feats: torch.Tensor,
        sub_feats: torch.Tensor,
        kinetic_feat: torch.Tensor,
    ) -> torch.Tensor:
        # post_feats/sub_feats: [B,T,C,H,W]
        if post_feats.ndim != 5:
            raise ValueError("post_feats must be [B,T,C,H,W]")
        if self.disabled:
            return post_feats.mean(dim=1)

        fused = []
        scores = []
        for idx in range(post_feats.shape[1]):
            post = post_feats[:, idx]
            sub = sub_feats[:, min(idx, sub_feats.shape[1] - 1)]
            residual = post - pre_feat
            gate = self.gate(torch.cat([residual, sub, kinetic_feat], dim=1))
            phase_fused = post + gate * residual
            fused.append(phase_fused)
            if self.aggregation == "attention":
                scores.append(self.attention_score(torch.cat([phase_fused, kinetic_feat], dim=1)))

        fused_stack = torch.stack(fused, dim=1)
        if self.aggregation == "attention" and len(scores) > 1:
            score_stack = torch.stack(scores, dim=1)
            weights = torch.softmax(score_stack, dim=1)
            return (fused_stack * weights).sum(dim=1)
        return fused_stack.mean(dim=1)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class BoundaryRefinementHead(nn.Module):
    def __init__(self, channels: int, disabled: bool = False) -> None:
        super().__init__()
        self.disabled = disabled
        self.boundary_feature = ConvBlock(channels, channels)
        self.boundary_logits = nn.Conv2d(channels, 1, kernel_size=1)
        self.attention = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, decoder_feature: torch.Tensor):
        if self.disabled:
            return decoder_feature, decoder_feature.new_zeros(
                (decoder_feature.shape[0], 1, decoder_feature.shape[2], decoder_feature.shape[3])
            )
        boundary_feature = self.boundary_feature(decoder_feature)
        boundary_logits = self.boundary_logits(boundary_feature)
        refined = decoder_feature + self.attention(boundary_feature) * boundary_feature
        return refined, boundary_logits


class SGKTFNet(nn.Module):
    """Subtraction-Guided Kinetic Temporal Fusion Network for 2D DCE-MRI."""

    def __init__(
        self,
        in_channels: int = 17,
        base_channels: int = 32,
        phase_indices: Optional[Dict] = None,
        compute_subtraction_if_missing: bool = True,
        use_kinetic_maps: bool = True,
        fusion_mode: str = "attention",
        use_boundary_head: bool = True,
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
        self.phase_indices = PhaseIndices.from_config(phase_indices)
        self.disable_kinetic_maps = bool(ablation.get("disable_kinetic_maps", False)) or not use_kinetic_maps
        self.disable_fusion = bool(ablation.get("disable_subtraction_guided_fusion", False))
        self.disable_boundary = bool(ablation.get("disable_boundary_head", False)) or not use_boundary_head

        self.kinetic_builder = EnhancementMapBuilder(
            phase_indices=phase_indices,
            compute_subtraction_if_missing=compute_subtraction_if_missing,
            eps=kinetic_eps,
            local_pool_kernel=local_pool_kernel,
            clip_value=kinetic_clip_value,
            normalize=kinetic_normalize,
            disable_kinetic_maps=self.disable_kinetic_maps,
        )
        self.phase_encoder = SharedPhaseEncoder(base_channels)
        self.sub_encoder = SharedPhaseEncoder(base_channels)
        self.kinetic_encoder = KineticEncoder(self.kinetic_builder.expected_channels, base_channels)

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.fusions = nn.ModuleList(
            [
                SubtractionGuidedResidualFusion(ch, aggregation=fusion_mode, disabled=self.disable_fusion)
                for ch in channels
            ]
        )
        self.up3 = UpBlock(channels[4], channels[3], channels[3])
        self.up2 = UpBlock(channels[3], channels[2], channels[2])
        self.up1 = UpBlock(channels[2], channels[1], channels[1])
        self.up0 = UpBlock(channels[1], channels[0], channels[0])
        self.boundary_head = BoundaryRefinementHead(channels[0], disabled=self.disable_boundary)
        self.seg_head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def _encode_phase_stack(self, stack: torch.Tensor, encoder: nn.Module) -> List[torch.Tensor]:
        b, t, h, w = stack.shape
        encoded = encoder(stack.reshape(b * t, 1, h, w))
        out = []
        for feat in encoded:
            out.append(feat.reshape(b, t, feat.shape[1], feat.shape[2], feat.shape[3]))
        return out

    def forward(self, x: torch.Tensor, return_dict: Optional[bool] = None):
        if x.ndim != 4:
            raise ValueError(f"SGKTFNet expects [B,C,H,W], got {tuple(x.shape)}")
        return_dict = self.return_dict if return_dict is None else return_dict

        phases = self.kinetic_builder.split_phases(x)
        pre = phases["pre"]
        post = phases["post"]
        subtraction = phases["subtraction"]
        kinetic_maps = self.kinetic_builder(x)

        pre_feats = self.phase_encoder(pre)
        post_feats = self._encode_phase_stack(post, self.phase_encoder)
        sub_feats = self._encode_phase_stack(subtraction, self.sub_encoder)
        kinetic_feats = self.kinetic_encoder(kinetic_maps)

        fused = []
        for level, fusion in enumerate(self.fusions):
            fused.append(fusion(pre_feats[level], post_feats[level], sub_feats[level], kinetic_feats[level]))

        x_dec = self.up3(fused[4], fused[3])
        x_dec = self.up2(x_dec, fused[2])
        x_dec = self.up1(x_dec, fused[1])
        x_dec = self.up0(x_dec, fused[0])
        refined, boundary_logits = self.boundary_head(x_dec)
        seg_logits = self.seg_head(refined)

        if not return_dict:
            return seg_logits
        return {
            "seg_logits": seg_logits,
            "logits": seg_logits,
            "boundary_logits": boundary_logits,
            "kinetic_maps": kinetic_maps,
            "debug": {
                "num_post_phases": post.shape[1],
                "num_subtraction_phases": subtraction.shape[1],
            },
        }


class SwinHR(SGKTFNet):
    """Compatibility wrapper for train_swinhr.py --model_name sg_ktfnet."""

    def __init__(self, in_channels: int = 1, attn_channels: int = 8, *args, **kwargs) -> None:
        total_channels = int(kwargs.pop("total_channels", in_channels + attn_channels))
        if total_channels >= 17:
            phase_indices = {"pre": 0, "post": list(range(1, 9)), "subtraction": list(range(9, 17))}
        else:
            phase_indices = {"pre": 0, "post": [], "subtraction": list(range(1, total_channels))}
        super().__init__(
            in_channels=total_channels,
            phase_indices=phase_indices,
            *args,
            **kwargs,
        )
