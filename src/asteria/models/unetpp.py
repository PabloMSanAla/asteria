"""
Nested U-Net (UNet++) Architecture for Astronomical Semantic Segmentation.
Pure PyTorch implementation with dense skip connections and deep supervision.
"""

from __future__ import annotations

from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard double convolution block:
    [Conv2d -> Norm -> GELU -> Conv2d -> Norm -> GELU]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        mid_channels: int | None = None,
    ):
        super().__init__()
        mid_channels = mid_channels or out_channels
        if norm_type == "group":
            groups1 = min(8, mid_channels)
            groups2 = min(8, out_channels)
            norm1 = nn.GroupNorm(groups1, mid_channels)
            norm2 = nn.GroupNorm(groups2, out_channels)
        else:
            norm1 = nn.BatchNorm2d(mid_channels)
            norm2 = nn.BatchNorm2d(out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm1,
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm2,
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Backward compatibility alias
VGGBlock = ConvBlock


class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) architecture for astronomical imaging.

    Implements dense skip pathways connecting encoder and decoder sub-networks,
    capturing multi-scale representations critical for separating faint, touching
    celestial sources from background noise and dense stellar cores.

    Reference:
        Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
        (DLMIA 2018 / IEEE TMI 2019).

    Parameters
    ----------
    in_channels : int, default 3
        Number of input channels (e.g. 3 for RGB or 3-channel stretched flux).
    num_classes : int, default 3
        Number of output classes (e.g. 0: Sky, 1: Source interior, 2: Boundary edge).
    deep_supervision : bool, default False
        If True, returns logits from all decoder sub-branches [X_0_1, X_0_2, X_0_3, X_0_4]
        during training.
    features : Tuple[int, ...], default (32, 64, 128, 256, 512)
        Feature map dimensions across hierarchical depths (i=0 to 4).
    norm_type : str, default 'batch'
        Normalization layer type ('batch' or 'group').
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        deep_supervision: bool = False,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        norm_type: str = "batch",
    ):
        super().__init__()
        if len(features) != 5:
            raise ValueError(f"UNetPlusPlus requires exactly 5 feature depths, got {len(features)}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.features = tuple(features)
        self.norm_type = norm_type

        nb_filter = list(features)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Backbone Encoder (j=0)
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0], norm_type=norm_type)
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], norm_type=norm_type)
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], norm_type=norm_type)
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], norm_type=norm_type)
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], norm_type=norm_type)

        # Nested Dense Skip Nodes (j=1)
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], norm_type=norm_type)
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], norm_type=norm_type)
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], norm_type=norm_type)
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3], norm_type=norm_type)

        # Nested Dense Skip Nodes (j=2)
        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], norm_type=norm_type)
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], norm_type=norm_type)
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], norm_type=norm_type)

        # Nested Dense Skip Nodes (j=3)
        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], norm_type=norm_type)
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], norm_type=norm_type)

        # Final Decoder Node (j=4)
        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], norm_type=norm_type)

        # Classification Heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Encoder column (j=0)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested Column j=1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # Nested Column j=2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Nested Column j=3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Nested Column j=4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            if self.training:
                return [out1, out2, out3, out4]
            else:
                return out4

        return self.final(x0_4)

