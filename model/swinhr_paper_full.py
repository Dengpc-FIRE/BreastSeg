import torch

from .swinhr_kinetic import KineticMapEncoder
from .swinhr_dcs import DcsSilhouetteSwinHR


class SwinHR(DcsSilhouetteSwinHR):
    """Paper-oriented full model: kinetic prior + reliability fusion + SDF head."""

    def __init__(self, *args, attn_channels=8, out_channels=1, **kwargs):
        super().__init__(
            *args,
            attn_channels=attn_channels,
            out_channels=2,
            use_curve_prior=True,
            use_reliability_fusion=True,
            use_combo_backbone=True,
            **kwargs,
        )
        self.kinetic = KineticMapEncoder(attn_channels)

    def forward(self, x: torch.Tensor):
        image = x[:, : self.in_channels, :, :]
        subtraction = x[:, self.in_channels : self.in_channels + self.attn_channels, :, :]
        subtraction, kinetic_maps = self.kinetic(subtraction)
        if self.use_curve_prior:
            subtraction, time_weight = self.curve_prior(subtraction)
            self.aux = {"time_weight": time_weight, "kinetic_maps": kinetic_maps}
        else:
            self.aux = {"kinetic_maps": kinetic_maps}

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
        out = self.out(up_full)
        return {"logits": out[:, :1], "sdf": out[:, 1:2]}
