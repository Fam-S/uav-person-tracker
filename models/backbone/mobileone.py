from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MobileOneStageOutput:
    low_level: torch.Tensor
    high_level: torch.Tensor


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileOneBlock(nn.Module):
    """Small readable approximation of an unfused MobileOne block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.use_skip = stride == 1 and in_channels == out_channels
        self.depthwise_3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.depthwise_1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depthwise = self.depthwise_3x3(x) + self.depthwise_1x1(x)
        if self.use_skip:
            depthwise = depthwise + x
        return self.activation(self.pointwise(depthwise))


class MobileOneS2Backbone(nn.Module):
    """Minimal MobileOne-S2-style backbone exposing stage-2 and stage-3 outputs."""

    def __init__(
        self,
        pretrained_path: str | None = None,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_input = bool(normalize_input)

        self.stem = ConvBNAct(3, 64, stride=2)
        self.stage1 = nn.Sequential(
            MobileOneBlock(64, 64, stride=1),
            MobileOneBlock(64, 128, stride=2),
        )
        self.stage2 = nn.Sequential(
            MobileOneBlock(128, 128, stride=1),
            MobileOneBlock(128, 256, stride=2),
        )
        self.stage3 = nn.Sequential(
            MobileOneBlock(256, 256, stride=1),
            MobileOneBlock(256, 512, stride=2),
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, pretrained_path: str) -> None:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {key.replace("module.", ""): value for key, value in state_dict.items()}
        self.load_state_dict(cleaned, strict=False)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.pixel_mean) / self.pixel_std

    def forward_features(self, x: torch.Tensor) -> MobileOneStageOutput:
        if self.normalize_input:
            x = self.normalize(x)
        x = self.stem(x)
        x = self.stage1(x)
        low_level = self.stage2(x)
        high_level = self.stage3(low_level)
        return MobileOneStageOutput(low_level=low_level, high_level=high_level)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward_features(x)
        return outputs.low_level, outputs.high_level
