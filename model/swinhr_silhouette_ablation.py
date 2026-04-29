from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetOutBlock
from monai.networks.blocks.convolutions import Convolution
from monai.utils import ensure_tuple_rep

from .swinhr_v9 import H_RLK, SwinTransformer


class MultiScaleInputStem(nn.Module):
    """Lightweight replacement stem before the Swin patch embedding."""

    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(torch.cat([x, self.local(x), self.context(x)], dim=1))


class TemporalSubtractionEncoder(nn.Module):
    """Model the 1-8 minute subtraction sequence before 2D feature extraction."""

    def __init__(self, channels: int):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(4),
            nn.GELU(),
            nn.Conv3d(4, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(1),
            nn.GELU(),
        )
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(1)
        temporal_feat = self.temporal(seq).squeeze(1)
        return x + self.channel_mix(temporal_feat)


class PriorInjectionBlock(nn.Module):
    """PIB: inject silhouette prior into image features with spatial reliability."""

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
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, image_feat: torch.Tensor, silhouette_feat: torch.Tensor) -> torch.Tensor:
        prior = self.prior(silhouette_feat)
        pair = torch.cat([image_feat, prior], dim=1)
        guided = image_feat + self.spatial_gate(pair) * self.channel_gate(pair) * prior
        return self.out(torch.cat([image_feat, guided], dim=1))


class RLKRefineBlock(nn.Module):
    """Large-kernel refinement after prior injection."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.rlk = H_RLK(
            dimensions=2,
            in_channels=channels,
            out_channels=channels,
            large_kernel=kernel_size,
            if_padding=1,
            strides=1,
            if_smallkernel=True,
        )
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x + self.rlk(x)))


class FusionBlock(nn.Module):
    def __init__(self, channels: int, use_rlk: bool = False, kernel_size: int = 7):
        super().__init__()
        self.pib = PriorInjectionBlock(channels)
        self.rlk = RLKRefineBlock(channels, kernel_size) if use_rlk else nn.Identity()

    def forward(self, image_feat: torch.Tensor, silhouette_feat: torch.Tensor) -> torch.Tensor:
        return self.rlk(self.pib(image_feat, silhouette_feat))


class DenseDecoderBlock(nn.Module):
    """Decoder replacement with dense skip fusion."""

    def __init__(self, spatial_dims: int, in_channels: int, skip_channels: int, out_channels: int, norm_name="instance"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            adn_ordering="NDA",
            norm=norm_name,
            act=("prelu", {"init": 0.2}),
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels * 3,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            adn_ordering="NDA",
            norm=norm_name,
            act=("prelu", {"init": 0.2}),
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(inp)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        x1 = self.conv1(torch.cat([up, skip], dim=1))
        return self.conv2(torch.cat([up, skip, x1], dim=1))


class EdgeAwareHead(nn.Module):
    """Small boundary refinement head used in the combined model."""

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )
        self.out = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x + self.edge(x))


class BaseSilhouetteSwinHR(nn.Module):
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
        use_input_stem: bool = False,
        use_temporal: bool = False,
        use_rlk_fusion: bool = False,
        use_dense_decoder: bool = False,
        use_edge_head: bool = False,
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
        self.image_stem = MultiScaleInputStem(in_channels) if use_input_stem else nn.Identity()
        self.silhouette_stem = MultiScaleInputStem(attn_channels) if use_input_stem else nn.Identity()
        self.temporal = TemporalSubtractionEncoder(attn_channels) if use_temporal else nn.Identity()

        encoder_kwargs = dict(
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
        self.image_encoder = SwinTransformer(in_chans=in_channels, **encoder_kwargs)
        self.silhouette_encoder = SwinTransformer(in_chans=attn_channels, **encoder_kwargs)

        self.fuse4 = FusionBlock(feature_size * 16, use_rlk=use_rlk_fusion, kernel_size=7)
        self.fuse3 = FusionBlock(feature_size * 8, use_rlk=use_rlk_fusion, kernel_size=9)
        self.fuse2 = FusionBlock(feature_size * 4, use_rlk=use_rlk_fusion, kernel_size=11)
        self.fuse1 = FusionBlock(feature_size * 2, use_rlk=use_rlk_fusion, kernel_size=13)
        self.fuse0 = FusionBlock(feature_size, use_rlk=use_rlk_fusion, kernel_size=13)
        self.fuse_full = FusionBlock(feature_size, use_rlk=use_rlk_fusion, kernel_size=13)

        decoder_cls = DenseDecoderBlock if use_dense_decoder else None
        if decoder_cls is None:
            from monai.networks.blocks import UnetrUpBlock

            self.decoder5 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=16 * feature_size, out_channels=8 * feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
            self.decoder4 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=8 * feature_size, out_channels=4 * feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
            self.decoder3 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=4 * feature_size, out_channels=2 * feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
            self.decoder2 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=2 * feature_size, out_channels=feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
            self.decoder1 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        else:
            self.decoder5 = decoder_cls(spatial_dims, 16 * feature_size, 8 * feature_size, 8 * feature_size, norm_name)
            self.decoder4 = decoder_cls(spatial_dims, 8 * feature_size, 4 * feature_size, 4 * feature_size, norm_name)
            self.decoder3 = decoder_cls(spatial_dims, 4 * feature_size, 2 * feature_size, 2 * feature_size, norm_name)
            self.decoder2 = decoder_cls(spatial_dims, 2 * feature_size, feature_size, feature_size, norm_name)
            self.decoder1 = decoder_cls(spatial_dims, feature_size, feature_size, feature_size, norm_name)

        self.out = EdgeAwareHead(feature_size, out_channels) if use_edge_head else UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x[:, : self.in_channels, :, :]
        silhouette = x[:, self.in_channels : self.in_channels + self.attn_channels, :, :]

        image = self.image_stem(image)
        silhouette = self.silhouette_stem(self.temporal(silhouette))

        i0, i1, i2, i3, i4, i_full = self.image_encoder(image)
        s0, s1, s2, s3, s4, s_full = self.silhouette_encoder(silhouette)

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
