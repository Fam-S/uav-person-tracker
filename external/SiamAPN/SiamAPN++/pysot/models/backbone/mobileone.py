from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_MOBILEONE_ROOT = Path(__file__).resolve().parents[5] / "ml-mobileone"
if str(_MOBILEONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOBILEONE_ROOT))

from mobileone import mobileone  # noqa: E402


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MobileOneS2Backbone(nn.Module):
    """MobileOne-S2 wrapper that preserves the original SiamAPN++ feature contract.

    Outputs:
        - feature_1: [B, 384, 8, 8] for template / [B, 384, 28, 28] for search
        - feature_2: [B, 256, 6, 6] for template / [B, 256, 26, 26] for search
    """

    def __init__(self):
        super(MobileOneS2Backbone, self).__init__()
        base_model = mobileone(variant="s2")

        # Register MobileOne stages directly so pretrained checkpoints from Apple
        # still line up with the wrapper state dict.
        self.stage0 = base_model.stage0
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.stage4 = base_model.stage4

        # Stage-2 gives the right resolution trend for SiamAPN++, while stage-3
        # adds stronger semantics. We upsample stage-3, fuse both, and then
        # reduce the spatial size back to the original AlexNet contract.
        self.fusion_proj = ConvBNReLU(256 + 640, 384, kernel_size=1)
        self.contract_reduce = ConvBNReLU(384, 384, kernel_size=9, stride=1, padding=0)
        self.head_proj = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
        )

        for param in self.stage0.parameters():
            param.requires_grad = False
        for param in self.stage1.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        stage2 = self.stage2(x)
        stage3 = self.stage3(stage2)

        stage3 = F.interpolate(stage3, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion_proj(torch.cat([stage2, stage3], dim=1))

        feature_1 = self.contract_reduce(fused)
        feature_2 = self.head_proj(feature_1)
        return feature_1, feature_2
