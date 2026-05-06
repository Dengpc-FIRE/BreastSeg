import torch
import torch.nn as nn

from .swinhr_dcs import DcsSilhouetteSwinHR


class KineticMapEncoder(nn.Module):
    """Convert SUB1-SUB8 into DCE kinetic prior maps."""

    def __init__(self, time_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(8, time_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(time_channels),
            nn.GELU(),
            nn.Conv2d(time_channels, time_channels, kernel_size=3, padding=1, groups=time_channels, bias=False),
            nn.InstanceNorm2d(time_channels),
            nn.GELU(),
            nn.Conv2d(time_channels, time_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(time_channels),
            nn.GELU(),
        )

    def forward(self, sub: torch.Tensor):
        b, t, h, w = sub.shape
        eps = 1e-6
        time = torch.linspace(0, 1, t, device=sub.device, dtype=sub.dtype).view(1, t, 1, 1)

        auc = sub.mean(dim=1, keepdim=True)
        peak, peak_idx = sub.max(dim=1, keepdim=True)
        ttp = peak_idx.to(sub.dtype) / max(t - 1, 1)
        early = sub[:, : max(1, t // 2)].mean(dim=1, keepdim=True)
        late = sub[:, max(1, t // 2) :].mean(dim=1, keepdim=True)
        wash_in = sub[:, 1:3].mean(dim=1, keepdim=True) - sub[:, :1]
        wash_out = sub[:, -1:] - peak
        variance = sub.var(dim=1, keepdim=True, unbiased=False)
        centroid = (sub.clamp_min(0) * time).sum(dim=1, keepdim=True) / (sub.clamp_min(0).sum(dim=1, keepdim=True) + eps)

        maps = torch.cat([auc, peak, ttp, early, late, wash_in, wash_out, variance + centroid], dim=1)
        kinetic = self.project(maps)
        return sub + kinetic, maps


class SwinHR(DcsSilhouetteSwinHR):
    """Kinetic-prior SwinHR using handcrafted DCE enhancement maps."""

    def __init__(self, *args, attn_channels=8, **kwargs):
        super().__init__(
            *args,
            attn_channels=attn_channels,
            use_curve_prior=False,
            use_reliability_fusion=True,
            use_combo_backbone=False,
            **kwargs,
        )
        self.kinetic = KineticMapEncoder(attn_channels)

    def forward(self, x: torch.Tensor):
        image = x[:, : self.in_channels, :, :]
        subtraction = x[:, self.in_channels : self.in_channels + self.attn_channels, :, :]
        subtraction, kinetic_maps = self.kinetic(subtraction)
        self.aux = {"kinetic_maps": kinetic_maps, "subtraction_response": subtraction.detach().abs().amax(dim=1, keepdim=True)}

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
        up3 = self.decoder5(inp=fused4, skip=fused3)
        up2 = self.decoder4(inp=up3, skip=fused2)
        up1 = self.decoder3(inp=up2, skip=fused1)
        up0 = self.decoder2(inp=up1, skip=fused0)
        up_full = self.decoder1(inp=up0, skip=fused_full)
        return self.out(up_full)
