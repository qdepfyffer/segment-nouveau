"""
FRRN-A (Full-Resolution Residual Network)
Follows Pohlen et al. 2017, "Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes"
Full-size (~17.7M parameters) network from the 2022 CIWA pistachio paper
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.padding import pad_to_multiple

# (out_channels, number of FRRUs, scale) per stage, from the FRRN-A layout in Pohlen et al. 2017
ENCODER_STAGES = [(96, 3, 2), (192, 4, 4), (384, 2, 8), (384, 2, 16)]
DECODER_STAGES = [(192, 2, 8), (192, 2, 4), (96, 2, 2)]


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    """
    Convolution, batch norm, ReLU - in a row, maintaining height and width
    :param in_channels:     Number of channels coming in
    :param out_channels:    Number of channels going out
    :param kernel_size:     Size of the (square) convolution kernel, defaults to 3
    :return:                The three layers as one module
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualUnit(nn.Module):
    """
    Two 3x3 convolutions whose result is added back onto the input (out = x + f(x))
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = conv_bn_relu(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.conv1(x)))


class FRRU(nn.Module):
    """
    Full-resolution residual unit - joins pooling stream (y) with the full-resolution residual stream (z)
    Returns an updated y at its own resolution and an updated z at full resolution
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int, residual_channels: int = 32) -> None:
        super().__init__()
        self.scale = scale
        self.convs = nn.Sequential(
            conv_bn_relu(in_channels + residual_channels, out_channels),
            conv_bn_relu(out_channels, out_channels),
        )
        self.to_residual = nn.Conv2d(out_channels, residual_channels, kernel_size=1)

    def forward(self, y, z):
        z_small = F.max_pool2d(z, kernel_size=self.scale)
        y = self.convs(torch.cat([y, z_small], dim=1))
        z = z + F.interpolate(self.to_residual(y), scale_factor=self.scale, mode="nearest")
        return y, z


class FRRUStage(nn.Module):
    """
    A run of FRRUs at one scale - only the first one changes the channel count
    """
    
    def __init__(self, in_channels: int, out_channels: int, count: int, scale: int) -> None:
        """
        :param in_channels:     Channels of the pooling stream coming into the stage
        :param out_channels:    Channels of the pooling stream inside and after the stage
        :param count:           Number of FRRUs in the stage
        :param scale:           How many times smaller the pooling stream is compared to full resolution
        """

        super().__init__()
        self.frrus = nn.ModuleList([FRRU(in_channels if i == 0 else out_channels, out_channels, scale) for i in range(count)])

    def forward(self, y, z):
        for frru in self.frrus:
            y, z = frru(y, z)
        return y, z

class FRRNA(nn.Module):
    """
    Full FRRN-A - full-resolution residual stream (z) alongside a pooling stream (y) that is shrunk to 1/16 and back
    """

    size_multiple = 16  # Pooling stream is halved four times so height and width must divide by 16

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Start with 5x5 conv and 3 residual units at full res, then 1x1 conv to begin residual stream
        self.stem = nn.Sequential(
            conv_bn_relu(3, 48, kernel_size=5),
            ResidualUnit(48),
            ResidualUnit(48),
            ResidualUnit(48),
        )
        self.split = nn.Conv2d(48, 32, kernel_size=1)

        # Encoder then decoder - "channels" tracks the width of the pooling stream from one stage to the next
        channels = 48

        self.encoder = nn.ModuleList()
        for out_channels, count, scale in ENCODER_STAGES:
            self.encoder.append(FRRUStage(channels, out_channels, count, scale))
            channels = out_channels

        self.decoder = nn.ModuleList()
        for out_channels, count, scale in DECODER_STAGES:
            self.decoder.append(FRRUStage(channels, out_channels, count, scale))
            channels = out_channels

        # Rejoin both streams at full resolution, three residual units, then one score per class
        self.merge = nn.Conv2d(channels + 32, 48, kernel_size=1)
        self.tail = nn.Sequential(ResidualUnit(48), ResidualUnit(48), ResidualUnit(48))
        self.head = nn.Conv2d(48, num_classes, kernel_size=1)

    def forward(self, images):
        """
        Defines the forward pass of the model
        :param images:  The images being shown to the model
        :return:        Logits of shape [batch_size, num_classes, height, width]
        """

        height, width = images.shape[-2:]
        x = self.stem(pad_to_multiple(images, self.size_multiple))
        y, z = x, self.split(x)

        for stage in self.encoder:
            y = F.max_pool2d(y, kernel_size=2)
            y, z = stage(y, z)

        for stage in self.decoder:
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            y, z = stage(y, z)


        y = F.interpolate(y, scale_factor=2, mode="nearest")
        x = self.merge(torch.cat([y, z], dim=1))
        logits = self.head(self.tail(x))
        return logits[..., :height, :width]