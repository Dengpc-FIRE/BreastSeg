import torch

from .swinhr_dcs import DcsSilhouetteSwinHR


class SwinHR(DcsSilhouetteSwinHR):
    """Segmentation + signed-distance geometry head."""

    def __init__(self, *args, out_channels=1, **kwargs):
        super().__init__(
            *args,
            out_channels=2,
            use_curve_prior=True,
            use_reliability_fusion=True,
            use_combo_backbone=True,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        return {"logits": out[:, :1], "sdf": out[:, 1:2]}
