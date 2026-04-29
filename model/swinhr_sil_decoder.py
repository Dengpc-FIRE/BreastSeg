from .swinhr_silhouette_ablation import BaseSilhouetteSwinHR


class SwinHR(BaseSilhouetteSwinHR):
    """Ablation D: dense decoder + PIB."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_dense_decoder=True, **kwargs)
