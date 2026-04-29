from .swinhr_silhouette_ablation import BaseSilhouetteSwinHR


class SwinHR(BaseSilhouetteSwinHR):
    """Ablation B: multi-scale input stem + PIB."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_input_stem=True, **kwargs)
