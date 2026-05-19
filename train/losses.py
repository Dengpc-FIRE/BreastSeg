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


def attention_smoothness_loss(attention_maps) -> torch.Tensor:
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
        ref = attention_maps[0]
        return ref.new_tensor(0.0)
    return torch.nan_to_num(torch.stack(losses).mean(), nan=0.0, posinf=0.0, neginf=0.0)


class KPTANetLoss(nn.Module):
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
        self.lambda_boundary = 0.0 if ablation.get("disable_boundary_head", False) else lambda_boundary
        self.lambda_uncertainty = 0.0 if ablation.get("disable_uncertainty_head", False) else lambda_uncertainty
        self.lambda_attention_smooth = (
            0.0 if ablation.get("disable_attention_smooth_loss", False) else lambda_attention_smooth
        )
        self.boundary_thickness = boundary_thickness
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, output, target: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        seg_logits, info = unpack_model_output(output)
        target = target.float()
        loss = self.seg_loss(seg_logits, target)
        boundary_target = boundary_target_2d(target, thickness=self.boundary_thickness)

        if self.lambda_boundary > 0 and "boundary_logits" in info:
            loss = loss + self.lambda_boundary * self.bce(info["boundary_logits"].float(), boundary_target)

        if self.lambda_uncertainty > 0 and "uncertainty_logits" in info:
            with torch.no_grad():
                error = (torch.sigmoid(seg_logits.detach()) - target).abs()
                uncertainty_target = (boundary_target + error).clamp(0, 1)
            loss = loss + self.lambda_uncertainty * self.bce(info["uncertainty_logits"].float(), uncertainty_target)

        if self.lambda_attention_smooth > 0:
            attention_maps = info.get("attention_maps", [])
            smooth = attention_smoothness_loss(attention_maps) if attention_maps else seg_logits.new_tensor(0.0)
            loss = loss + self.lambda_attention_smooth * smooth

        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


class TemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, min_tumor_pixels: int = 20) -> None:
        super().__init__()
        self.temperature = temperature
        self.min_tumor_pixels = min_tumor_pixels

    def forward(self, embeddings, target: torch.Tensor) -> torch.Tensor:
        if not embeddings or "phase_features" not in embeddings:
            return target.new_tensor(0.0)
        phase_features = embeddings["phase_features"].float()
        phase_mask = embeddings.get("phase_mask")
        if phase_features.ndim != 5 or phase_features.shape[1] <= 1:
            return phase_features.new_tensor(0.0)
        if phase_mask is None:
            phase_mask = phase_features.new_ones((phase_features.shape[0], phase_features.shape[1]))
        phase_mask = phase_mask.to(device=phase_features.device, dtype=phase_features.dtype)
        target_small = F.interpolate(target.float(), size=phase_features.shape[-2:], mode="nearest")

        tumor_embeds = []
        bg_embeds = []
        sample_ids = []
        for b in range(phase_features.shape[0]):
            tumor = target_small[b : b + 1]
            bg = 1.0 - tumor
            if tumor.sum() < self.min_tumor_pixels or bg.sum() < self.min_tumor_pixels:
                continue
            available = torch.nonzero(phase_mask[b] > 0.5, as_tuple=False).flatten()
            if available.numel() <= 1:
                continue
            for t in available.tolist():
                feat = phase_features[b, t]
                tumor_embeds.append(self._masked_pool(feat, tumor[0]))
                bg_embeds.append(self._masked_pool(feat, bg[0]))
                sample_ids.append(b)

        if len(tumor_embeds) < 2 or len(bg_embeds) < 1:
            return phase_features.new_tensor(0.0)

        tumor_embeds = F.normalize(torch.stack(tumor_embeds), dim=1)
        bg_embeds = F.normalize(torch.stack(bg_embeds), dim=1)
        sample_ids_t = torch.tensor(sample_ids, device=phase_features.device)
        losses = []
        for idx in range(tumor_embeds.shape[0]):
            pos_mask = (sample_ids_t == sample_ids_t[idx])
            pos_mask[idx] = False
            if pos_mask.sum() < 1:
                continue
            pos_logits = torch.matmul(tumor_embeds[idx : idx + 1], tumor_embeds[pos_mask].T).flatten() / self.temperature
            neg_logits_bg = torch.matmul(tumor_embeds[idx : idx + 1], bg_embeds.T).flatten() / self.temperature
            other_tumor = sample_ids_t != sample_ids_t[idx]
            neg_logits = neg_logits_bg
            if other_tumor.any():
                neg_logits_other = torch.matmul(tumor_embeds[idx : idx + 1], tumor_embeds[other_tumor].T).flatten() / self.temperature
                neg_logits = torch.cat([neg_logits, neg_logits_other], dim=0)
            numerator = torch.logsumexp(pos_logits, dim=0)
            denominator = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=0), dim=0)
            losses.append(-(numerator - denominator))

        if not losses:
            return phase_features.new_tensor(0.0)
        return torch.nan_to_num(torch.stack(losses).mean(), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _masked_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (feat * mask).sum(dim=(1, 2)) / denom


class KPRNetLoss(nn.Module):
    def __init__(
        self,
        lambda_contrastive: float = 0.1,
        lambda_kinetic: float = 0.05,
        contrastive_temperature: float = 0.1,
        min_tumor_pixels: int = 20,
        kinetic_margin: float = 0.05,
        peritumor_ring_size: int = 5,
        kinetic_eps: float = 1e-6,
        bce_weight: float = 0.5,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        ablation = ablation or {}
        self.seg_loss = DiceBCELoss(bce_weight=bce_weight)
        self.lambda_contrastive = (
            0.0 if ablation.get("disable_temporal_contrastive_loss", False) else lambda_contrastive
        )
        self.lambda_kinetic = 0.0 if ablation.get("disable_kinetic_loss", False) else lambda_kinetic
        self.kinetic_loss = KineticConsistencyLoss(
            margin=kinetic_margin,
            ring_size=peritumor_ring_size,
            eps=kinetic_eps,
        )
        self.temporal_contrastive = TemporalContrastiveLoss(
            temperature=contrastive_temperature,
            min_tumor_pixels=min_tumor_pixels,
        )

    def forward(self, output, target: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        seg_logits, info = unpack_model_output(output)
        loss = self.seg_loss(seg_logits, target)
        if self.lambda_kinetic > 0:
            loss = loss + self.lambda_kinetic * self.kinetic_loss(seg_logits, info.get("kinetic_maps"), target)
        if self.lambda_contrastive > 0:
            loss = loss + self.lambda_contrastive * self.temporal_contrastive(
                info.get("contrastive_embeddings"),
                target,
            )
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
