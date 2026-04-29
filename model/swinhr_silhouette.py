from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import UnetOutBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep

from .swinhr_v9 import SwinTransformer


class SilhouetteGuidedFusion(nn.Module):
    """
    Fuse image features with subtraction/silhouette features.

    The image branch keeps texture and boundary details. The silhouette branch
    provides a coarse location and shape prior from post-contrast subtraction.
    A learned reliability gate decides how much silhouette information should
    be injected at each spatial location.
    """

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 8)

        self.silhouette_prior = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

        self.reliability_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, image_feat: torch.Tensor, silhouette_feat: torch.Tensor) -> torch.Tensor:
        prior = self.silhouette_prior(silhouette_feat)
        gate = self.reliability_gate(torch.cat([image_feat, prior], dim=1))
        guided = image_feat + gate * prior
        return self.refine(torch.cat([image_feat, guided], dim=1))


class SwinHR(nn.Module):
    """
    Silhouette-guided SwinHR for BreastDM segmentation.

    Expected input layout:
        x[:, 0:1] -> original image branch
        x[:, 1:]  -> subtraction/silhouette branch

    The default setting matches the current 9-channel training pipeline:
    1 original image channel + 8 subtraction channels.
    """

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
    ) -> None:
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        self.in_channels = in_channels
        self.attn_channels = attn_channels

        self.image_encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        self.silhouette_encoder = SwinTransformer(
            in_chans=attn_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        self.fuse4 = SilhouetteGuidedFusion(feature_size * 16)
        self.fuse3 = SilhouetteGuidedFusion(feature_size * 8)
        self.fuse2 = SilhouetteGuidedFusion(feature_size * 4)
        self.fuse1 = SilhouetteGuidedFusion(feature_size * 2)
        self.fuse0 = SilhouetteGuidedFusion(feature_size)
        self.fuse_full = SilhouetteGuidedFusion(feature_size)

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x[:, : self.in_channels, :, :]
        silhouette = x[:, self.in_channels : self.in_channels + self.attn_channels, :, :]

        image_feats = self.image_encoder(image)
        silhouette_feats = self.silhouette_encoder(silhouette)

        i0, i1, i2, i3, i4, i_full = image_feats
        s0, s1, s2, s3, s4, s_full = silhouette_feats

        fused4 = self.fuse4(i4, s4)
        fused3 = self.fuse3(i3, s3)
        fused2 = self.fuse2(i2, s2)
        fused1 = self.fuse1(i1, s1)
        fused0 = self.fuse0(i0, s0)
        fused_full = self.fuse_full(i_full, s_full)

        up3 = self.decoder5(inp=fused4, skip=fused3)
        up2 = self.decoder4(inp=up3, skip=fused2)
        up1 = self.decoder3(inp=up2, skip=fused1)
        up0 = self.decoder2(inp=up1, skip=fused0)
        up_full = self.decoder1(inp=up0, skip=fused_full)
        return self.out(up_full)
