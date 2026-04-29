from .swinhr_silhouette_ablation import BaseSilhouetteSwinHR


class SwinHR(BaseSilhouetteSwinHR):
    """Full model: stem + temporal modeling + RLK fusion + dense decoder + edge head."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            use_input_stem=True,
            use_temporal=True,
            use_rlk_fusion=True,
            use_dense_decoder=True,
            use_edge_head=True,
            **kwargs,
        )
