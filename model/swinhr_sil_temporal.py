from .swinhr_silhouette_ablation import BaseSilhouetteSwinHR


class SwinHR(BaseSilhouetteSwinHR):
    """Ablation C: temporal subtraction modeling + PIB."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_temporal=True, **kwargs)
