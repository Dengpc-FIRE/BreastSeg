import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1e-6,
        dice_reduction: str = "batch",
        bce_pos_weight=None,
        auto_pos_weight: bool = False,
        max_pos_weight: float = 50.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.dice_reduction = dice_reduction
        self.bce_pos_weight = bce_pos_weight
        self.auto_pos_weight = bool(auto_pos_weight)
        self.max_pos_weight = float(max_pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        if self.dice_reduction == "sample":
            dims = tuple(range(1, probs.ndim))
            intersection = (probs * target).sum(dim=dims)
            denom = probs.sum(dim=dims) + target.sum(dim=dims)
            dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (denom + self.smooth)).mean()
        else:
            intersection = (probs * target).sum()
            dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        pos_weight = self._pos_weight(target, logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

    def _pos_weight(self, target: torch.Tensor, device):
        if self.auto_pos_weight:
            pos = target.sum().clamp_min(1.0)
            neg = (target.numel() - target.sum()).clamp_min(1.0)
            value = (neg / pos).clamp(max=self.max_pos_weight)
            return value.detach().to(device).view(1)
        if self.bce_pos_weight is None:
            return None
        return torch.tensor([float(self.bce_pos_weight)], device=device)
