from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsampling block with a U-Net skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)

        # Handles images whose dimensions are not exactly divisible by 16.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Four-level 2D U-Net that returns unnormalized class logits."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernels: int = 8,
        factor: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # With the existing SEGTHOR settings this becomes 16 channels.
        base_channels = max(16, kernels * factor)

        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = Down(base_channels, base_channels * 2)
        self.encoder3 = Down(base_channels * 2, base_channels * 4)
        self.encoder4 = Down(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            Down(base_channels * 8, base_channels * 16),
            nn.Dropout2d(p=dropout),
        )

        self.decoder1 = Up(base_channels * 16, base_channels * 8)
        self.decoder2 = Up(base_channels * 8, base_channels * 4)
        self.decoder3 = Up(base_channels * 4, base_channels * 2)
        self.decoder4 = Up(base_channels * 2, base_channels)

        self.output = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1,
        )

    def init_weights(self) -> None:
        """Initialize weights using the same callable interface as ENet."""

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)

        x = self.bottleneck(skip4)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        return self.output(x)