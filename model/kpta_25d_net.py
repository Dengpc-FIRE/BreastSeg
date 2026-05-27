from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dce_kinetic_utils import PhaseIndices
from .sg_ktfnet import ConvBlock, UpBlock


class KineticMapBuilder25D(nn.Module):
    """Build slice-preserving pseudo-kinetic maps from raw 2.5D DCE input."""

    def __init__(
        self,
        phase_indices: Optional[Dict] = None,
        kinetic_maps: Optional[Sequence[str]] = None,
        eps: float = 1e-6,
        clip_value: Optional[float] = None,
        normalize_after_kinetic: bool = True,
        normalize_kinetic: bool = True,
    ) -> None:
        super().__init__()
        self.phase_indices = PhaseIndices.from_config(phase_indices)
        self.kinetic_maps = list(
            [
                "sub_stack",
                "peak_enhancement",
                "mean_enhancement",
                "temporal_std",
                "early_enhancement",
                "late_enhancement",
            ]
            if kinetic_maps is None
            else kinetic_maps
        )
        self.eps = eps
        self.clip_value = clip_value
        self.normalize_after_kinetic = normalize_after_kinetic
        self.normalize_kinetic = normalize_kinetic

    @property
    def expected_channels(self) -> int:
        n_sub = max(len(self.phase_indices.subtraction), len(self.phase_indices.post), 1)
        channels = 0
        for name in self.kinetic_maps:
            if name in {"sub_stack", "relative_enhancement"}:
                channels += n_sub
            else:
                channels += 1
        return max(channels, 1)

    def split(self, x: torch.Tensor):
        pre = x[:, :, self.phase_indices.pre : self.phase_indices.pre + 1]
        post = self._select(x, self.phase_indices.post)
        sub = self._select(x, self.phase_indices.subtraction)
        if post is None and sub is not None:
            post = pre + sub
        if sub is None and post is not None:
            sub = post - pre
        if post is None and sub is None:
            post = pre
            sub = torch.zeros_like(pre)
        assert post is not None and sub is not None
        return pre, post, sub

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_after_kinetic:
            return x.float()
        x = x.float()
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        std = x.std(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)

    def normalize_kinetic_maps(self, kinetic_maps: torch.Tensor) -> torch.Tensor:
        if not self.normalize_kinetic:
            return kinetic_maps.float()
        kinetic_maps = kinetic_maps.float()
        mean = kinetic_maps.mean(dim=(1, 2, 3, 4), keepdim=True)
        std = kinetic_maps.std(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
        return (kinetic_maps - mean) / (std + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,K,T,H,W], got {tuple(x.shape)}")
        pre, post, sub = self.split(x.float())
        maps = []
        for name in self.kinetic_maps:
            if name == "sub_stack":
                maps.append(sub)
            elif name == "peak_enhancement":
                maps.append(sub.amax(dim=2, keepdim=True))
            elif name == "mean_enhancement":
                maps.append(sub.mean(dim=2, keepdim=True))
            elif name == "temporal_std":
                maps.append(post.std(dim=2, keepdim=True, unbiased=False))
            elif name == "early_enhancement":
                maps.append(sub[:, :, :1])
            elif name == "late_enhancement":
                maps.append(sub[:, :, -1:])
            elif name == "relative_enhancement":
                maps.append(sub / (pre.abs() + self.eps))
            else:
                raise ValueError(f"Unknown 2.5D kinetic map: {name}")
        kinetic = torch.cat(maps, dim=2) if maps else x.new_zeros((x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4]))
        if self.clip_value is not None and self.clip_value > 0:
            kinetic = kinetic.clamp(-self.clip_value, self.clip_value)
        kinetic = torch.nan_to_num(kinetic, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(self.normalize_kinetic_maps(kinetic), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _select(x: torch.Tensor, indices) -> Optional[torch.Tensor]:
        valid = [idx for idx in indices if 0 <= idx < x.shape[2]]
        if not valid:
            return None
        return x[:, :, valid]


class SliceWiseCNNStem(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.stem = ConvBlock(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, t, h, w = x.shape
        feat = self.stem(x.reshape(b * k * t, 1, h, w))
        return feat.reshape(b, k, t, feat.shape[1], feat.shape[2], feat.shape[3])


class SliceContextAggregation(nn.Module):
    def __init__(self, channels: int, disabled: bool = False, use_center_residual: bool = True) -> None:
        super().__init__()
        self.disabled = disabled
        self.use_center_residual = use_center_residual
        self.score = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: [B,K,T,C,H,W] -> [B,T,C,H,W]
        if self.disabled or x.shape[1] == 1:
            center = x[:, x.shape[1] // 2]
            attn = x.new_ones((x.shape[0], x.shape[1], x.shape[2], 1, x.shape[-2], x.shape[-1])) / float(x.shape[1])
            return center, attn
        b, k, t, c, h, w = x.shape
        scores = self.score(x.reshape(b * k * t, c, h, w)).reshape(b, k, t, 1, h, w)
        weights = torch.softmax(scores, dim=1)
        aggregated = (x * weights).sum(dim=1)
        if self.use_center_residual:
            center = x[:, k // 2]
            aggregated = 0.5 * (aggregated + center)
        return aggregated, weights


class KineticPriorBranch25D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, disabled_slice_context: bool = False) -> None:
        super().__init__()
        self.encoder = ConvBlock(in_channels, out_channels)
        self.slice_agg = SliceContextAggregation(out_channels, disabled=disabled_slice_context)

    def forward(self, kinetic_maps: torch.Tensor):
        b, k, m, h, w = kinetic_maps.shape
        feat = self.encoder(kinetic_maps.reshape(b * k, m, h, w))
        feat = feat.reshape(b, k, 1, feat.shape[1], feat.shape[2], feat.shape[3])
        aggregated, attn = self.slice_agg(feat)
        return aggregated[:, 0], attn


class PixelWisePhaseAttention25D(nn.Module):
    def __init__(self, channels: int, num_phases: int, disabled: bool = False) -> None:
        super().__init__()
        self.disabled = disabled
        self.num_phases = num_phases
        self.score = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, phase_feats: torch.Tensor, kinetic_feat: torch.Tensor):
        # phase_feats: [B,T,C,H,W]
        b, t, c, h, w = phase_feats.shape
        if self.disabled or t == 1:
            attn = phase_feats.new_full((b, t, 1, h, w), 1.0 / float(t))
            return phase_feats.mean(dim=1), attn
        kinetic = kinetic_feat.unsqueeze(1).expand(-1, t, -1, -1, -1)
        logits = self.score(torch.cat([phase_feats, kinetic], dim=2).reshape(b * t, c * 2, h, w))
        logits = logits.reshape(b, t, 1, h, w).squeeze(2)
        attn = torch.softmax(logits, dim=1).unsqueeze(2)
        return (phase_feats * attn).sum(dim=1), attn


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowSelfAttention(nn.Module):
    def __init__(self, channels: int, window_size: int = 7, num_heads: int = 4) -> None:
        super().__init__()
        heads = max(1, min(num_heads, channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.channels = channels
        self.window_size = window_size
        self.num_heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        return self.proj(out)


class SwinWindowBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 7,
        num_heads: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowSelfAttention(channels, window_size=window_size, num_heads=num_heads)
        hidden = int(channels * mlp_ratio)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x.shape[1], x.shape[2]
        shift = self.shift_size if min(hp, wp) > self.window_size else 0
        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        attn_mask = self._shifted_window_mask(hp, wp, x.device, x.dtype) if shift > 0 else None
        windows = _window_partition(self.norm1(x), self.window_size)
        windows = self.attn(windows, mask=attn_mask)
        x = _window_reverse(windows, self.window_size, hp, wp, b)
        if shift > 0:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        if pad_h or pad_w:
            x = x[:, :h, :w, :].contiguous()
        x = shortcut + x.permute(0, 3, 1, 2).contiguous()

        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.mlp(self.norm2(x))
        return shortcut + x.permute(0, 3, 1, 2).contiguous()

    def _shifted_window_mask(self, height: int, width: int, device, dtype) -> torch.Tensor:
        img_mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class LightweightSwinBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 7,
        disabled: bool = False,
    ) -> None:
        super().__init__()
        self.disabled = disabled
        window_size = max(2, int(window_size))
        self.blocks = nn.ModuleList(
            [
                SwinWindowBlock(
                    channels,
                    window_size=window_size,
                    num_heads=num_heads,
                    shift_size=0 if idx % 2 == 0 else window_size // 2,
                )
                for idx in range(max(1, depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return x
        for block in self.blocks:
            x = block(x)
        return x


class HybridEncoder25D(nn.Module):
    def __init__(
        self,
        channels: int,
        transformer_depth: int = 2,
        num_heads: int = 4,
        window_size: int = 7,
        disable_transformer: bool = False,
    ) -> None:
        super().__init__()
        self.down1 = ConvBlock(channels, channels * 2)
        self.down2 = ConvBlock(channels * 2, channels * 4)
        self.down3 = ConvBlock(channels * 4, channels * 8)
        self.bottleneck = ConvBlock(channels * 8, channels * 16)
        self.transformer = LightweightSwinBottleneck(
            channels * 16,
            depth=transformer_depth,
            num_heads=num_heads,
            window_size=window_size,
            disabled=disable_transformer,
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x0: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.down1(self.pool(x0))
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))
        x4 = self.transformer(x4)
        return [x0, x1, x2, x3, x4]


class UncertaintyBoundaryRefinement25D(nn.Module):
    def __init__(self, channels: int, disabled: bool = False) -> None:
        super().__init__()
        self.disabled = disabled
        self.boundary_feature = ConvBlock(channels, channels)
        self.boundary_gate = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, decoder_feature: torch.Tensor, uncertainty_map: torch.Tensor):
        boundary_feature = self.boundary_feature(decoder_feature)
        if self.disabled:
            return decoder_feature, boundary_feature
        refined = decoder_feature + uncertainty_map * self.boundary_gate(boundary_feature) * boundary_feature
        return refined, boundary_feature


class KPTA25DNet(nn.Module):
    """Slice-aware 2.5D KPTA-Net with kinetic phase attention."""

    def __init__(
        self,
        in_phases: int = 17,
        num_slices: int = 3,
        base_channels: int = 32,
        phase_indices: Optional[Dict] = None,
        kinetic_maps: Optional[Sequence[str]] = None,
        normalize_after_kinetic: bool = True,
        normalize_kinetic: bool = True,
        slice_context_mode: str = "attention",
        phase_attention: bool = True,
        use_uncertainty_head: bool = True,
        use_boundary_head: bool = True,
        use_uncertainty_refinement: bool = True,
        hybrid_encoder: Optional[Dict] = None,
        return_dict: bool = True,
        kinetic_eps: float = 1e-6,
        kinetic_clip_value: Optional[float] = None,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        ablation = ablation or {}
        hybrid_encoder = hybrid_encoder or {}
        self.in_phases = in_phases
        self.num_slices = num_slices
        self.return_dict = return_dict
        self.disable_kinetic_maps = bool(ablation.get("disable_kinetic_maps", False))
        self.disable_slice_context = bool(ablation.get("disable_slice_context", False)) or slice_context_mode == "none"
        self.disable_phase_attention = bool(ablation.get("disable_pixelwise_phase_attention", False)) or not phase_attention
        self.disable_transformer = bool(ablation.get("disable_transformer_bottleneck", False))
        self.disable_boundary = bool(ablation.get("disable_boundary_head", False)) or not use_boundary_head
        self.disable_uncertainty = bool(ablation.get("disable_uncertainty_head", False)) or not use_uncertainty_head
        self.disable_refinement = bool(ablation.get("disable_uncertainty_refinement", False)) or not use_uncertainty_refinement

        self.kinetic_builder = KineticMapBuilder25D(
            phase_indices=phase_indices,
            kinetic_maps=[] if self.disable_kinetic_maps else kinetic_maps,
            eps=kinetic_eps,
            clip_value=kinetic_clip_value,
            normalize_after_kinetic=normalize_after_kinetic,
            normalize_kinetic=normalize_kinetic,
        )
        self.slice_stem = SliceWiseCNNStem(base_channels)
        self.slice_agg = SliceContextAggregation(base_channels, disabled=self.disable_slice_context)
        self.kinetic_branch = KineticPriorBranch25D(
            self.kinetic_builder.expected_channels,
            base_channels,
            disabled_slice_context=self.disable_slice_context,
        )
        self.phase_attention = PixelWisePhaseAttention25D(base_channels, in_phases, disabled=self.disable_phase_attention)
        self.hybrid_encoder = HybridEncoder25D(
            base_channels,
            transformer_depth=int(hybrid_encoder.get("transformer_depth", 2)),
            num_heads=int(hybrid_encoder.get("num_heads", 4)),
            window_size=int(hybrid_encoder.get("window_size", 7)),
            disable_transformer=self.disable_transformer,
        )
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.up3 = UpBlock(channels[4], channels[3], channels[3])
        self.up2 = UpBlock(channels[3], channels[2], channels[2])
        self.up1 = UpBlock(channels[2], channels[1], channels[1])
        self.up0 = UpBlock(channels[1], channels[0], channels[0])
        self.coarse_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.refinement = UncertaintyBoundaryRefinement25D(channels[0], disabled=self.disable_refinement)
        self.boundary_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.seg_head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_dict: Optional[bool] = None):
        if x.ndim != 5:
            raise ValueError(f"KPTA25DNet expects [B,K,T,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_slices:
            raise ValueError(f"KPTA25DNet expected {self.num_slices} slices, got {x.shape[1]}")
        if x.shape[2] != self.in_phases:
            raise ValueError(f"KPTA25DNet expected {self.in_phases} phases, got {x.shape[2]}")
        return_dict = self.return_dict if return_dict is None else return_dict

        kinetic_maps = self.kinetic_builder(x)
        x_norm = self.kinetic_builder.normalize_input(x)
        slice_phase_features = self.slice_stem(x_norm)
        context_features, slice_attention_maps = self.slice_agg(slice_phase_features)
        kinetic_feature, kinetic_slice_attention = self.kinetic_branch(kinetic_maps)
        fused0, phase_attention_maps = self.phase_attention(context_features, kinetic_feature)

        skips = self.hybrid_encoder(fused0)
        dec = self.up3(skips[4], skips[3])
        dec = self.up2(dec, skips[2])
        dec = self.up1(dec, skips[1])
        dec = self.up0(dec, skips[0])

        coarse_logits = self.coarse_head(dec)
        coarse_prob = torch.sigmoid(coarse_logits.float())
        uncertainty_prob = (1.0 - torch.abs(2.0 * coarse_prob - 1.0)).clamp(1e-3, 1.0 - 1e-3)
        uncertainty_logits = torch.logit(uncertainty_prob).to(dtype=coarse_logits.dtype)
        uncertainty_map = uncertainty_prob.to(dtype=dec.dtype)
        if self.disable_uncertainty:
            uncertainty_logits = torch.zeros_like(uncertainty_logits)
            uncertainty_map = torch.zeros_like(uncertainty_map)
        refined, boundary_feature = self.refinement(dec, uncertainty_map)
        boundary_logits = self.boundary_head(boundary_feature)
        if self.disable_boundary:
            boundary_logits = torch.zeros_like(boundary_logits)
        seg_logits = self.seg_head(refined)

        if not return_dict:
            return seg_logits
        return {
            "seg_logits": seg_logits,
            "logits": seg_logits,
            "coarse_logits": coarse_logits,
            "boundary_logits": boundary_logits,
            "uncertainty_logits": uncertainty_logits,
            "uncertainty_map": uncertainty_map,
            "kinetic_maps": kinetic_maps,
            "phase_attention_maps": [phase_attention_maps],
            "attention_maps": [phase_attention_maps],
            "slice_attention_maps": [slice_attention_maps, kinetic_slice_attention],
            "debug": {
                "num_slices": x.shape[1],
                "num_phases": x.shape[2],
                "num_kinetic_maps": kinetic_maps.shape[2],
            },
        }


class SwinHR(KPTA25DNet):
    """Compatibility wrapper for dynamic loading by model name."""

    pass
