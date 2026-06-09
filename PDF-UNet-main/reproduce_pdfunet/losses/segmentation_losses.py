import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1e-6) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        inter = torch.sum(probs * target, dim=dims)
        denom = torch.sum(probs + target, dim=dims)
        dice_loss = 1.0 - torch.mean((2.0 * inter + self.smooth) / (denom + self.smooth))
        return self.dice_weight * dice_loss + self.bce_weight * self.bce(logits, target)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        tp = torch.sum(probs * target, dim=dims)
        fp = torch.sum(probs * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - probs) * target, dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.mean(torch.pow(1.0 - tversky, self.gamma))


def build_loss(cfg) -> nn.Module:
    name = cfg["loss"]["name"].lower()
    if name == "focaltversky":
        return FocalTverskyLoss(cfg["loss"]["alpha"], cfg["loss"]["beta"], cfg["loss"]["gamma"])
    if name == "dicebce":
        return DiceBCELoss(cfg["loss"]["dice_weight"], cfg["loss"]["bce_weight"])
    raise ValueError(f"Unsupported loss: {cfg['loss']['name']}")

