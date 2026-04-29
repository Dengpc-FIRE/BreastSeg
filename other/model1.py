"""
HybridTemporalNet (model1.py)

Dual-Stream Spatio-Temporal Network for 9-channel Breast DCE-MRI.

Key components:
- TABlock: Tumor-Aware Multi-Scale Block (3 branches: 3x3, two 3x3 (as 5x5), 1x1), concat + 1x1 + residual
- Encoder: stack of TABlocks with downsampling; this encoder is reused (weight-sharing)
  for both the static stream (V0) and every frame of the dynamic stream (V_seq).
- TSFF: attention-based Temporal Series Feature Fusion that uses static feature as
  query to compute attention over time for each spatial location, then aggregates
  the dynamic features accordingly and adds to static feature (residual fusion).
- Decoder: U-Net style decoder receiving fused skip features at each level.

Notes about weight sharing and temporal fusion are added as comments in the code.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TABlock(nn.Module):
    """Tumor-Aware (TA) Block: multi-scale branches and residual connection.

    Branches:
      - 3x3 conv
      - two 3x3 convs (approx 5x5 receptive field)
      - 1x1 conv
    Fuse: concat -> 1x1 conv to reduce channels
    Residual: added if input/output channels match
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = out_ch // 2

        # Branch 1: 3x3
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # Branch 2: two 3x3 (approx 5x5)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 1x1
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # Fusion 1x1 to reduce channels to out_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 3, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # if in_ch != out_ch, provide a projection for residual
        if in_ch != out_ch:
            self.res_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.res_proj = None

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)

        cat = torch.cat([b1, b2, b3], dim=1)
        out = self.fuse(cat)

        if self.res_proj is not None:
            res = self.res_proj(x)
        else:
            res = x

        return F.relu(out + res)


class Encoder(nn.Module):
    """Encoder composed of TABlocks. This encoder is meant to be weight-shared.

    Weight sharing: we will instantiate a single `Encoder` and call it twice:
      - once for static input V0 (shape [B,1,H,W])
      - once for the dynamic super-batch (shape [B*T,1,H,W])

    The outputs at each stage are collected for skip connections.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth)]

        layers = []
        prev_ch = in_ch
        for out_ch in chs:
            layers.append(nn.Sequential(
                TABlock(prev_ch, out_ch),
            ))
            prev_ch = out_ch

        self.layers = nn.ModuleList(layers)
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(depth - 1)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        out = x
        for i in range(self.depth):
            out = self.layers[i](out)
            feats.append(out)
            if i < self.depth - 1:
                out = self.pools[i](out)
        return feats  # list of features from shallow->deep


class TSFF(nn.Module):
    """Temporal Series Feature Fusion (TSFF)

    Fuse dynamic features F_seq (B,T,C,H,W) guided by static feature f0 (B,C,H,W).

    Mechanism:
      - Project static and dynamic features to a lower-dim embedding (C') via 1x1 conv.
      - Compute compatibility score per time and spatial location: dot over C'.
      - Softmax over time dimension for each (b,h,w) -> attn[b,t,h,w].
      - Weighted sum of dynamic features over t using attn maps -> dynamic_agg[b,c,h,w].
      - Fuse: out = f0 + dynamic_agg (residual). Optionally a 1x1 conv mixes channels.
    """

    def __init__(self, ch: int, embed_ch: int = None):
        super().__init__()
        if embed_ch is None:
            embed_ch = max(8, ch // 4)
        self.key_conv = nn.Conv2d(ch, embed_ch, kernel_size=1, bias=False)
        self.query_conv = nn.Conv2d(ch, embed_ch, kernel_size=1, bias=False)
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, f_static: torch.Tensor, F_seq: torch.Tensor) -> torch.Tensor:
        # f_static: [B, C, H, W]
        # F_seq: [B, T, C, H, W]
        B, T, C, H, W = F_seq.shape

        # project
        q = self.query_conv(f_static)  # [B, C', H, W]
        # project dynamic: merge batch and time for conv
        F_flat = F_seq.view(B * T, C, H, W)
        k_flat = self.key_conv(F_flat)  # [B*T, C', H, W]
        Ck = k_flat.shape[1]
        k = k_flat.view(B, T, Ck, H, W)  # [B,T,C',H,W]

        # compute compatibility: dot product over channel dim -> [B,T,H,W]
        # q: [B,C',H,W], k: [B,T,C',H,W]
        # compute q * k summed over C'
        q_exp = q.unsqueeze(1)  # [B,1,C',H,W]
        attn_logits = (q_exp * k).sum(dim=2)  # [B,T,H,W]

        # softmax over time dim -> attn weights
        attn = torch.softmax(attn_logits, dim=1)  # [B,T,H,W]

        # weight dynamic features
        attn_exp = attn.unsqueeze(2)  # [B,T,1,H,W]
        weighted = F_seq * attn_exp  # [B,T,C,H,W]
        dynamic_agg = weighted.sum(dim=1)  # [B,C,H,W]

        # fuse
        out = f_static + dynamic_agg
        out = self.out_conv(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = TABlock(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HybridTemporalNet(nn.Module):
    """Main model.

    Weight-sharing: `self.encoder` is a single Encoder instance used for both
    the static input and for each dynamic frame (by reshaping dynamic inputs
    to a super-batch of shape [B*T,1,H,W]). This forces the network to learn
    per-frame appearance features with identical weights.
    """

    def __init__(self, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.base_ch = base_ch
        self.depth = depth

        # shared encoder (weight-shared across frames)
        self.encoder = Encoder(in_ch=1, base_ch=base_ch, depth=depth)

        # TSFF modules per level
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.tsff_modules = nn.ModuleList([TSFF(ch) for ch in chs])

        # decoder upblocks (reverse order)
        self.up3 = UpBlock(chs[-1], chs[-2])
        self.up2 = UpBlock(chs[-2], chs[-3])
        self.up1 = UpBlock(chs[-3], chs[-4]) if depth >= 4 else None

        # final conv
        self.final = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,9,H,W]
        B = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]

        # Static stream: V0
        v0 = x[:, 0:1, :, :]  # [B,1,H,W]
        # weight-sharing: encode static
        feats_static = self.encoder(v0)  # list [f1,f2,f3,f4]

        # Dynamic stream: V_seq (channels 1..8)
        vseq = x[:, 1:, :, :]  # [B,8,H,W]
        T = vseq.shape[1]

        # reshape to super-batch [B*T,1,H,W]
        vseq_flat = vseq.reshape(B * T, 1, H, W)

        # encode dynamic frames with the SAME encoder (weight-sharing)
        feats_seq_flat = self.encoder(vseq_flat)  # list of features per frame: each [B*T, C, H', W']

        # reshape each feature to [B, T, C, H', W'] and apply TSFF with corresponding static feature
        fused_feats = []
        for lvl, feat_flat in enumerate(feats_seq_flat):
            C = feat_flat.shape[1]
            Hf = feat_flat.shape[2]
            Wf = feat_flat.shape[3]
            feat_bt = feat_flat.view(B, T, C, Hf, Wf)

            # corresponding static feature at this level (may need spatial downsample due to pooling)
            f_static = feats_static[lvl]

            # TSFF: fuse dynamic sequence guided by static feature
            fused = self.tsff_modules[lvl](f_static, feat_bt)  # [B,C,H',W']
            fused_feats.append(fused)

        # Decoder: start from deepest fused feature
        x_dec = fused_feats[-1]
        x_dec = self.up3(x_dec, fused_feats[-2])
        x_dec = self.up2(x_dec, fused_feats[-3])
        if self.up1 is not None:
            x_dec = self.up1(x_dec, fused_feats[-4])

        logits = self.final(x_dec)
        # return logits (no sigmoid) -> training should use BCEWithLogitsLoss
        return logits


if __name__ == '__main__':
    # quick sanity check
    model = HybridTemporalNet(base_ch=16, depth=4)
    x = torch.randn(2, 9, 256, 256)
    y = model(x)
    print('Input:', x.shape, 'Output:', y.shape)
