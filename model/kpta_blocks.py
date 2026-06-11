"""Shared neural-network blocks used by the KPTA model variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class KineticEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        self.enc0 = ConvBlock(in_channels, channels[0])
        self.enc1 = ConvBlock(channels[0], channels[1])
        self.enc2 = ConvBlock(channels[1], channels[2])
        self.enc3 = ConvBlock(channels[2], channels[3])
        self.bottleneck = ConvBlock(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))
        return [x0, x1, x2, x3, x4]


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
