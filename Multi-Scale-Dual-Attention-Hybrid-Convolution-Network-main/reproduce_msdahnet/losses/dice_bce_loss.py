import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1e-6) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        intersection = (probs * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        bce_loss = self.bce(logits, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
