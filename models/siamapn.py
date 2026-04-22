from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import MobileOneS2Backbone


class DepthwiseCrossCorrelation(nn.Module):
    def forward(self, template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        if template.ndim != 4 or search.ndim != 4:
            raise ValueError("Expected 4D feature maps for cross-correlation.")
        if template.shape[:2] != search.shape[:2]:
            raise ValueError(
                f"Template/search feature mismatch: {tuple(template.shape)} vs {tuple(search.shape)}"
            )

        batch_size, channels, _, _ = template.shape
        search_h, search_w = search.shape[-2:]
        kernel_h, kernel_w = template.shape[-2:]

        search_reshaped = search.reshape(1, batch_size * channels, search_h, search_w)
        template_reshaped = template.reshape(batch_size * channels, 1, kernel_h, kernel_w)
        response = F.conv2d(search_reshaped, template_reshaped, groups=batch_size * channels)
        return response.reshape(batch_size, channels, response.shape[-2], response.shape[-1])


class FeatureAlign(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SiamAPNppMobileOne(nn.Module):
    """Minimal dual-level SiamAPN-style tracker."""

    def __init__(
        self,
        feature_channels: int = 192,
        pretrained_path: str | None = None,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = MobileOneS2Backbone(
            pretrained_path=pretrained_path,
            normalize_input=normalize_input,
        )
        self.low_align = FeatureAlign(256, feature_channels)
        self.high_align = FeatureAlign(512, feature_channels)
        self.correlation = DepthwiseCrossCorrelation()
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels, 4, kernel_size=1),
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low_level, high_level = self.backbone(x)
        return self.low_align(low_level), self.high_align(high_level)

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> dict[str, torch.Tensor]:
        if template.ndim != 4 or search.ndim != 4:
            raise ValueError("Expected template/search tensors with shape [B, C, H, W].")
        if template.size(0) != search.size(0):
            raise ValueError("Template and search batch sizes must match.")

        template_low, template_high = self._encode(template)
        search_low, search_high = self._encode(search)

        low_response = self.correlation(template_low, search_low)
        high_response = self.correlation(template_high, search_high)
        high_response = F.interpolate(
            high_response,
            size=low_response.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused = self.fusion(torch.cat([low_response, high_response], dim=1))
        bbox_pred = self.reg_head(fused).flatten(1)
        bbox_pred = torch.cat([bbox_pred[:, :2], F.softplus(bbox_pred[:, 2:])], dim=1)
        return {
            "bbox_pred": bbox_pred,
        }
