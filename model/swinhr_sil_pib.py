from .swinhr_silhouette_ablation import BaseSilhouetteSwinHR


class SwinHR(BaseSilhouetteSwinHR):
    """Ablation A: silhouette prior injection block only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
