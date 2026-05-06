from .swinhr_dcs import DcsSilhouetteSwinHR


class SwinHR(DcsSilhouetteSwinHR):
    """Ablation: dynamic enhancement curve prior only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curve_prior=True, use_reliability_fusion=False, use_combo_backbone=False, **kwargs)
