from .swinhr_dcs import DcsSilhouetteSwinHR


class SwinHR(DcsSilhouetteSwinHR):
    """Full model: curve prior + reliability fusion + combo backbone."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_curve_prior=True, use_reliability_fusion=True, use_combo_backbone=True, **kwargs)
