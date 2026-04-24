from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone._mobileone import mobileone, reparameterize_model


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileOneS2Backbone(nn.Module):
    """MobileOne-S2 wrapper with the original SiamAPN++ feature contract.

    For 127 templates it returns [B, 384, 8, 8] and [B, 256, 6, 6].
    For 287 searches it returns [B, 384, 28, 28] and [B, 256, 26, 26].
    """

    def __init__(
        self,
        pretrained_path: str | None = None,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_input = bool(normalize_input)

        base = mobileone(variant="s2", inference_mode=False)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            base.load_state_dict(state, strict=True)

        self.stage0 = base.stage0
        self.stage1 = base.stage1
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        del base

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

        self._low_channels = 384
        self._high_channels = 256

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    @property
    def low_channels(self) -> int:
        return self._low_channels

    @property
    def high_channels(self) -> int:
        return self._high_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_input:
            x = (x - self.pixel_mean) / self.pixel_std
        x = self.stage0(x)
        x = self.stage1(x)
        stage2 = self.stage2(x)
        stage3 = self.stage3(stage2)
        stage3 = F.interpolate(stage3, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion_proj(torch.cat([stage2, stage3], dim=1))
        feature_1 = self.contract_reduce(fused)
        feature_2 = self.head_proj(feature_1)
        return feature_1, feature_2

    def reparameterize(self) -> None:
        self.stage0 = reparameterize_model(self.stage0)
        self.stage1 = reparameterize_model(self.stage1)
        self.stage2 = reparameterize_model(self.stage2)
        self.stage3 = reparameterize_model(self.stage3)
        self.stage4 = reparameterize_model(self.stage4)
