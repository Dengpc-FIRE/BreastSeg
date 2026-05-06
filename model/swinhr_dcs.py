from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swinhr_silhouette_ablation import BaseSilhouetteSwinHR


class DynamicCurvePrior(nn.Module):
    """Encode SUB1-SUB8 as an enhancement curve instead of plain channels."""

    def __init__(self, time_channels: int):
        super().__init__()
        self.temporal_context = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(8),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(1),
            nn.GELU(),
        )
        self.time_score = nn.Sequential(
            nn.Conv2d(time_channels * 3, time_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(time_channels),
            nn.GELU(),
            nn.Conv2d(time_channels, time_channels, kernel_size=1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(time_channels, time_channels, kernel_size=3, padding=1, groups=time_channels, bias=False),
            nn.InstanceNorm2d(time_channels),
            nn.GELU(),
            nn.Conv2d(time_channels, time_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(time_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        # x: B,T,H,W. The temporal dimension is the enhancement curve.
        temporal = self.temporal_context(x.unsqueeze(1)).squeeze(1)
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        descriptor = torch.cat([x, temporal, delta], dim=1)
        time_weight = torch.softmax(self.time_score(descriptor), dim=1)
        weighted = x * time_weight
        return self.out(weighted + temporal), time_weight


class ReliabilityAwareFusion(nn.Module):
    """Fuse original-image and subtraction features with learned reliability."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.prior = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )
        self.spatial_reliability = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_reliability = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, image_feat: torch.Tensor, sub_feat: torch.Tensor):
        prior = self.prior(sub_feat)
        diff = torch.abs(image_feat - prior)
        pair = torch.cat([image_feat, prior, diff], dim=1)
        spatial = self.spatial_reliability(pair)
        channel = self.channel_reliability(pair)
        guided = image_feat + spatial * channel * prior
        return self.out(torch.cat([image_feat, guided], dim=1)), spatial


class DcsSilhouetteSwinHR(BaseSilhouetteSwinHR):
    """Dynamic Curve and reliability-guided Silhouette SwinHR."""

    def __init__(
        self,
        img_size: Union[Sequence[int], int] = (256, 256),
        in_channels: int = 1,
        attn_channels: int = 8,
        out_channels: int = 1,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        use_curve_prior: bool = True,
        use_reliability_fusion: bool = True,
        use_combo_backbone: bool = True,
    ) -> None:
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            attn_channels=attn_channels,
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            use_input_stem=use_combo_backbone,
            use_temporal=False,
            use_rlk_fusion=use_combo_backbone,
            use_dense_decoder=use_combo_backbone,
            use_edge_head=use_combo_backbone,
        )
        self.use_curve_prior = use_curve_prior
        self.use_reliability_fusion = use_reliability_fusion
        self.curve_prior = DynamicCurvePrior(attn_channels) if use_curve_prior else nn.Identity()

        if use_reliability_fusion:
            self.fuse4 = ReliabilityAwareFusion(feature_size * 16)
            self.fuse3 = ReliabilityAwareFusion(feature_size * 8)
            self.fuse2 = ReliabilityAwareFusion(feature_size * 4)
            self.fuse1 = ReliabilityAwareFusion(feature_size * 2)
            self.fuse0 = ReliabilityAwareFusion(feature_size)
            self.fuse_full = ReliabilityAwareFusion(feature_size)

        self.aux = {}

    def _fuse(self, block, image_feat, sub_feat, name):
        if self.use_reliability_fusion:
            fused, reliability = block(image_feat, sub_feat)
            self.aux[f"reliability_{name}"] = reliability
            return fused
        return block(image_feat, sub_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x[:, : self.in_channels, :, :]
        subtraction = x[:, self.in_channels : self.in_channels + self.attn_channels, :, :]

        if self.use_curve_prior:
            subtraction, time_weight = self.curve_prior(subtraction)
            self.aux = {"time_weight": time_weight}
        else:
            self.aux = {}

        image = self.image_stem(image)
        subtraction = self.silhouette_stem(subtraction)

        i0, i1, i2, i3, i4, i_full = self.image_encoder(image)
        s0, s1, s2, s3, s4, s_full = self.silhouette_encoder(subtraction)

        fused4 = self._fuse(self.fuse4, i4, s4, "4")
        fused3 = self._fuse(self.fuse3, i3, s3, "3")
        fused2 = self._fuse(self.fuse2, i2, s2, "2")
        fused1 = self._fuse(self.fuse1, i1, s1, "1")
        fused0 = self._fuse(self.fuse0, i0, s0, "0")
        fused_full = self._fuse(self.fuse_full, i_full, s_full, "full")

        self.aux["separability_feature"] = fused1
        self.aux["subtraction_response"] = subtraction.detach().abs().amax(dim=1, keepdim=True)

        up3 = self.decoder5(inp=fused4, skip=fused3)
        up2 = self.decoder4(inp=up3, skip=fused2)
        up1 = self.decoder3(inp=up2, skip=fused1)
        up0 = self.decoder2(inp=up1, skip=fused0)
        up_full = self.decoder1(inp=up0, skip=fused_full)
        return self.out(up_full)


class SwinHR(DcsSilhouetteSwinHR):
    pass
