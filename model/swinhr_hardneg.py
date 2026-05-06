from .swinhr_dcs import DcsSilhouetteSwinHR


class SwinHR(DcsSilhouetteSwinHR):
    """Model entry for enhancement-aware hard negative suppression experiments."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            use_curve_prior=True,
            use_reliability_fusion=True,
            use_combo_backbone=False,
            **kwargs,
        )
