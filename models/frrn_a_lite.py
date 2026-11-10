"""
FRRN-A lite - reduced FRRN-A-style network from the 2026 CIWA tomato paper
Keeps FRRN's two streams, pooling twice instead of four times, using one convolution per FRRU, and a 48 channel residual stream
For the full network see frrn_a.py
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.frrn_a import conv_bn_relu
from models.padding import pad_to_multiple


class PreActResidualUnit(nn.Module):
    """
    Residual unit with batch norm and ReLU before each convolution (out = x + f(x)) - x passes through untouched
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out


class LiteFRRU(nn.Module):
    """
    Single convolution FRRU - joins pooling stream (y) with full-resolution residual stream (z)
    Resizes with bilinear interpolation to exact sizes, works at any pair of resolutions
    """

    def __init__(self, in_channels: int, out_channels: int, residual_channels: int) -> None:
        super().__init__()
        self.merge = conv_bn_relu(in_channels + residual_channels, out_channels)
        self.to_residual = nn.Conv2d(out_channels, residual_channels, kernel_size=1, bias=False)

    def forward(self, y, z):
        z_small = F.interpolate(z, size=y.shape[-2:], mode="bilinear", align_corners=False)
        y = self.merge(torch.cat([y, z_small], dim=1))
        z = z + F.interpolate(self.to_residual(y), size=z.shape[-2:], mode="bilinear", align_corners=False)
        return y, z


class FRRNALite(nn.Module):
    """
    Lite version of the full FRRN-A implementation (as used in the 2026 CIWA tomato paper)
    Two streams like in the full implementation, but pools twice (instead of four times), uses one convolution per FRRU, and has <base> channels (48 by default) as defined in frrn_a_lite.yaml
    """

    size_multiple = 4

    def __init__(self, num_classes: int, base: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu(3, base), PreActResidualUnit(base), PreActResidualUnit(base))
        self.down1 = LiteFRRU(base, 2 * base, residual_channels=base)
        self.down2 = LiteFRRU(2 * base, 4 * base, residual_channels=base)
        self.bottleneck = nn.Sequential(conv_bn_relu(4 * base, 4 * base), conv_bn_relu(4 * base, 4 * base))
        self.up1 = nn.ConvTranspose2d(4 * base, 2 * base, kernel_size=2, stride=2, bias=False)
        self.frru_up1 = LiteFRRU(2 * base, 2 * base, residual_channels=base)
        self.up0 = nn.ConvTranspose2d(2 * base, base, kernel_size=2, stride=2, bias=False)
        self.frru_up0 = LiteFRRU(base, base, residual_channels=base)
        self.tail = nn.Sequential(PreActResidualUnit(base), PreActResidualUnit(base))
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        z = self.stem(pad_to_multiple(images, self.size_multiple))
        y, z = self.down1(F.max_pool2d(z, kernel_size=2), z)
        y, z = self.down2(F.max_pool2d(y, kernel_size=2), z)
        y = self.bottleneck(y)
        y, z = self.frru_up1(self.up1(y), z)
        y, z = self.frru_up0(self.up0(y), z)
        logits = self.head(self.tail(z))
        return logits[..., :height, :width]
    