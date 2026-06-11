"""Loss functions shared by the 2D and 2.5D KPTA models."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.kpta_utils import boundary_target_2d


def unpack_model_output(output) -> Tuple[torch.Tensor, Dict]:
    """Return segmentation logits and the complete auxiliary-output mapping."""
    if isinstance(output, dict):
        if "seg_logits" in output:
            return output["seg_logits"], output
        if "logits" in output:
            return output["logits"], output
        raise KeyError("KPTA model output must contain 'seg_logits' or 'logits'.")
    if isinstance(output, (tuple, list)):
        return output[0], {"extra": output[1:]}
    return output, {}


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (denominator + self.smooth)
        ).mean()
        if self.bce_weight <= 0:
            return dice
        return dice + self.bce_weight * self.bce(logits, target)


def attention_smoothness_loss(attention_maps) -> torch.Tensor:
    """Total-variation regularization for pixel-wise phase attention."""
    if not attention_maps:
        return torch.tensor(0.0)
    losses = []
    for attention in attention_maps:
        if attention.shape[1] <= 1:
            continue
        dy = (attention[..., 1:, :] - attention[..., :-1, :]).abs().mean()
        dx = (attention[..., :, 1:] - attention[..., :, :-1]).abs().mean()
        losses.append(dy + dx)
    if not losses:
        return attention_maps[0].new_tensor(0.0)
    return torch.nan_to_num(
        torch.stack(losses).mean(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


class KPTANetLoss(nn.Module):
    """Segmentation, boundary, uncertainty, and attention losses for KPTA."""

    def __init__(
        self,
        lambda_boundary: float = 0.2,
        lambda_uncertainty: float = 0.1,
        lambda_attention_smooth: float = 0.01,
        boundary_thickness: int = 3,
        bce_weight: float = 0.5,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        ablation = ablation or {}
        self.seg_loss = DiceBCELoss(bce_weight=bce_weight)
        self.lambda_boundary = (
            0.0
            if ablation.get("disable_boundary_head", False)
            else lambda_boundary
        )
        self.lambda_uncertainty = (
            0.0
            if ablation.get("disable_uncertainty_head", False)
            else lambda_uncertainty
        )
        self.lambda_attention_smooth = (
            0.0
            if ablation.get("disable_attention_smooth_loss", False)
            else lambda_attention_smooth
        )
        self.boundary_thickness = boundary_thickness
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        output,
        target: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del images
        seg_logits, info = unpack_model_output(output)
        target = target.float()
        loss = self.seg_loss(seg_logits, target)
        boundary_target = boundary_target_2d(
            target,
            thickness=self.boundary_thickness,
        )

        if self.lambda_boundary > 0 and "boundary_logits" in info:
            loss = loss + self.lambda_boundary * self.bce(
                info["boundary_logits"].float(),
                boundary_target,
            )

        if self.lambda_uncertainty > 0 and "uncertainty_logits" in info:
            with torch.no_grad():
                error = (torch.sigmoid(seg_logits.detach()) - target).abs()
                uncertainty_target = (boundary_target + error).clamp(0, 1)
            loss = loss + self.lambda_uncertainty * self.bce(
                info["uncertainty_logits"].float(),
                uncertainty_target,
            )

        if self.lambda_attention_smooth > 0:
            attention_maps = info.get("attention_maps", [])
            smoothness = (
                attention_smoothness_loss(attention_maps)
                if attention_maps
                else seg_logits.new_tensor(0.0)
            )
            loss = loss + self.lambda_attention_smooth * smoothness

        return loss


class KPTA25DNetLoss(KPTANetLoss):
    """The 2.5D model exposes the same supervised heads as the 2D model."""
