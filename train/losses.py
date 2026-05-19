from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dce_kinetic_utils import boundary_target_2d


def unpack_model_output(output) -> Tuple[torch.Tensor, Dict]:
    if isinstance(output, dict):
        if "seg_logits" in output:
            return output["seg_logits"], output
        if "logits" in output:
            return output["logits"], output
        raise KeyError("Dictionary model output must contain 'seg_logits' or 'logits'.")
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
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denom + self.smooth)).mean()
        if self.bce_weight <= 0:
            return dice
        return dice + self.bce_weight * self.bce(logits, target)


class KineticConsistencyLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.05,
        ring_size: int = 5,
        eps: float = 1e-6,
        use_predicted_mask: bool = False,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.ring_size = ring_size
        self.eps = eps
        self.use_predicted_mask = use_predicted_mask

    def forward(
        self,
        seg_logits: torch.Tensor,
        kinetic_maps: Optional[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if kinetic_maps is None or kinetic_maps.numel() == 0:
            return seg_logits.new_tensor(0.0)

        if self.use_predicted_mask:
            tumor = (torch.sigmoid(seg_logits.detach()) > 0.5).float()
        else:
            tumor = target.float().clamp(0, 1)

        if tumor.sum() < 1:
            return seg_logits.new_tensor(0.0)

        ring = (F.max_pool2d(tumor, kernel_size=self._kernel_size(), stride=1, padding=self._kernel_size() // 2) - tumor).clamp(0, 1)
        if ring.sum() < 1:
            return seg_logits.new_tensor(0.0)

        maps = kinetic_maps.float().abs()
        tumor_sum = tumor.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        ring_sum = ring.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        e_in = (maps * tumor).sum(dim=(2, 3), keepdim=True) / tumor_sum
        e_out = (maps * ring).sum(dim=(2, 3), keepdim=True) / ring_sum
        loss = F.relu(self.margin - e_in + e_out)
        return torch.nan_to_num(loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)

    def _kernel_size(self) -> int:
        kernel = max(3, int(self.ring_size))
        if kernel % 2 == 0:
            kernel += 1
        return kernel


class SGKTFNetLoss(nn.Module):
    def __init__(
        self,
        lambda_boundary: float = 0.2,
        lambda_kinetic: float = 0.1,
        kinetic_margin: float = 0.05,
        boundary_thickness: int = 3,
        peritumor_ring_size: int = 5,
        kinetic_eps: float = 1e-6,
        bce_weight: float = 0.5,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        ablation = ablation or {}
        self.seg_loss = DiceBCELoss(bce_weight=bce_weight)
        self.lambda_boundary = 0.0 if ablation.get("disable_boundary_head", False) else lambda_boundary
        self.lambda_kinetic = 0.0 if ablation.get("disable_kinetic_loss", False) else lambda_kinetic
        self.boundary_thickness = boundary_thickness
        self.boundary_bce = nn.BCEWithLogitsLoss()
        self.kinetic_loss = KineticConsistencyLoss(
            margin=kinetic_margin,
            ring_size=peritumor_ring_size,
            eps=kinetic_eps,
        )

    def forward(self, output, target: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        seg_logits, info = unpack_model_output(output)
        loss = self.seg_loss(seg_logits, target)

        if self.lambda_boundary > 0 and "boundary_logits" in info:
            boundary_logits = info["boundary_logits"]
            boundary_target = boundary_target_2d(target, thickness=self.boundary_thickness)
            loss = loss + self.lambda_boundary * self.boundary_bce(boundary_logits.float(), boundary_target.float())

        if self.lambda_kinetic > 0:
            loss = loss + self.lambda_kinetic * self.kinetic_loss(
                seg_logits,
                info.get("kinetic_maps"),
                target,
            )
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
