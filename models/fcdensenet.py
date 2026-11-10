"""
FC-DenseNet103 ("Tiramisu") - Jégou et al. 2017, "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
Matches the ~25.7M parameter version used in the 2026 CIWA tomato paper - transition-up layers upsample every channel and up-path blocks keep their full concatenation
"""

import torch
from torch import nn

from models.padding import pad_to_multiple

GROWTH_RATE = 16
STEM_CHANNELS = 48
DOWN_LAYERS = (4, 5, 7, 10, 12)
BOTTLENECK_LAYERS = 15
UP_LAYERS = (12, 10, 7, 5, 4)


def bn_relu_conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    """
    Batch norm, ReLU, convolution - in a row
    :param in_channels:     Number of channels coming in
    :param out_channels:    Number of channels going out
    :param kernel_size:     Size of the (square) convolution kernel, defaults to 3
    :return:                The three layers as one module
    """

    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
    )


class DenseBlock(nn.Module):
    """
    Each layer adds growth_rate new channels, computed from everything before it
    """

    def __init__(self, in_channels: int, num_layers: int, growth_rate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [bn_relu_conv(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]
        )
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x


def transition_down(channels: int) -> nn.Sequential:
    """
    Halves resolution while keeping the channel count
    """

    return nn.Sequential(bn_relu_conv(channels, channels, kernel_size=1), nn.MaxPool2d(2))


class FCDenseNet103(nn.Module):
    """
    U-shaped network of dense blocks with skip connections; see module docstring for how it differs from the paper.
    """

    size_multiple = 32

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, STEM_CHANNELS, kernel_size=3, padding=1, bias=False)

        channels = STEM_CHANNELS
        skip_channels = []
        self.down_blocks = nn.ModuleList()
        self.transitions_down = nn.ModuleList()
        for num_layers in DOWN_LAYERS:
            block = DenseBlock(channels, num_layers, GROWTH_RATE)
            self.down_blocks.append(block)
            channels = block.out_channels
            skip_channels.append(channels)
            self.transitions_down.append(transition_down(channels))

        self.bottleneck = DenseBlock(channels, BOTTLENECK_LAYERS, GROWTH_RATE)
        channels = self.bottleneck.out_channels

        self.transitions_up = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for num_layers, skip in zip(UP_LAYERS, reversed(skip_channels)):
            self.transitions_up.append(
                nn.ConvTranspose2d(channels, skip, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            )
            block = DenseBlock(2 * skip, num_layers, GROWTH_RATE)
            self.up_blocks.append(block)
            channels = block.out_channels

        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        x = self.stem(pad_to_multiple(images, self.size_multiple))

        skips = []
        for block, down in zip(self.down_blocks, self.transitions_down):
            x = block(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.transitions_up, self.up_blocks, reversed(skips)):
            x = block(torch.cat([up(x), skip], dim=1))

        logits = self.head(x)
        return logits[..., :height, :width]
