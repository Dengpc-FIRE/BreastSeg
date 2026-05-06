from .swinhr_dcs import DcsSilhouetteSwinHR


class SwinHR(DcsSilhouetteSwinHR):
    """Ablation: reliability-aware fusion only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curve_prior=False, use_reliability_fusion=True, use_combo_backbone=False, **kwargs)
