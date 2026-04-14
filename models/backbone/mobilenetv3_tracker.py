from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class BackboneOutput:
    features: torch.Tensor
    stride: int


def _resolve_mobilenet_weights(variant: str, pretrained: bool):
    if not pretrained:
        return None

    variant = variant.lower()
    if variant == "small":
        if hasattr(models, "MobileNet_V3_Small_Weights"):
            return models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        return "DEFAULT"

    if variant == "large":
        if hasattr(models, "MobileNet_V3_Large_Weights"):
            return models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        return "DEFAULT"

    raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")


def _build_mobilenet_features(variant: str, pretrained: bool) -> nn.Sequential:
    variant = variant.lower()
    weights = _resolve_mobilenet_weights(variant, pretrained)

    if variant == "small":
        model = models.mobilenet_v3_small(weights=weights)
    elif variant == "large":
        model = models.mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")

    return model.features


def _infer_feature_channels(features: nn.Sequential) -> int:
    last_conv_out = None
    for module in reversed(list(features.modules())):
        if isinstance(module, nn.Conv2d):
            last_conv_out = module.out_channels
            break

    if last_conv_out is None:
        raise ValueError("Could not infer output channels from MobileNetV3 features.")

    return int(last_conv_out)


class ConvProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3 feature extractor for Siamese tracking.

    Input:
        x: [B, 3, H, W] where H/W can be 127, 255, or other valid sizes.

    Output:
        Tensor of shape [B, out_channels, Hf, Wf]

    Notes:
    - Uses shared weights for template/search branches by reusing the same module.
    - Applies ImageNet normalization internally.
    - Removes classification head by keeping only `model.features`.
    - Adds a 1x1 projection to force a fixed channel dimension.
    """

    def __init__(
        self,
        variant: str = "small",
        pretrained: bool = True,
        out_channels: int = 96,
        trainable: bool = False,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()

        self.variant = variant.lower()
        self.pretrained = bool(pretrained)
        self.out_channels = int(out_channels)
        self.normalize_input = bool(normalize_input)

        self.features = _build_mobilenet_features(
            variant=self.variant,
            pretrained=self.pretrained,
        )

        in_channels = _infer_feature_channels(self.features)
        self.projection = ConvProjection(in_channels=in_channels, out_channels=self.out_channels)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        self.output_stride = 32
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool) -> None:
        for param in self.features.parameters():
            param.requires_grad = trainable

    def freeze(self) -> None:
        self.set_trainable(False)

    def unfreeze(self) -> None:
        self.set_trainable(True)

    def unfreeze_last_n_feature_blocks(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be at least 1.")

        for param in self.features.parameters():
            param.requires_grad = False

        child_blocks = list(self.features.children())
        for block in child_blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape [B, C, H, W], got {tuple(x.shape)}")
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 input channels, got {x.size(1)}")

        return (x - self.pixel_mean) / self.pixel_std

    def forward_features(self, x: torch.Tensor) -> BackboneOutput:
        if self.normalize_input:
            x = self.normalize(x)

        x = self.features(x)
        x = self.projection(x)

        return BackboneOutput(features=x, stride=self.output_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x).features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = MobileNetV3Backbone(
        variant="small",
        pretrained=True,
        out_channels=96,
        trainable=False,
        normalize_input=True,
    ).to(device)

    template = torch.randn(2, 3, 127, 127, device=device)
    search = torch.randn(2, 3, 255, 255, device=device)

    z_feat = backbone(template)
    x_feat = backbone(search)

    print("template input:", template.shape)
    print("search input:", search.shape)
    print("template features:", z_feat.shape)
    print("search features:", x_feat.shape)