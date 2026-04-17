from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import MobileNetV3Backbone


# ─── Depthwise Cross-Correlation ─────────────────────────────────────────────

@dataclass(frozen=True)
class CorrelationOutput:
    response: torch.Tensor


class DepthwiseCrossCorrelation(nn.Module):
    """
    Depthwise cross-correlation for Siamese tracking.

    Inputs:
        z: template features [B, C, Hz, Wz]
        x: search features   [B, C, Hx, Wx]

    Output:
        response map         [B, C, Hr, Wr]

    Notes:
    - Uses grouped convolution to perform per-channel correlation.
    - Each sample in the batch is handled independently.
    - This is the common lightweight choice for Siamese trackers.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4 or x.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors [B, C, H, W], got z={tuple(z.shape)}, x={tuple(x.shape)}"
            )

        bz, cz, hz, wz = z.shape
        bx, cx, hx, wx = x.shape

        if bz != bx:
            raise ValueError(f"Batch size mismatch: z batch={bz}, x batch={bx}")
        if cz != cx:
            raise ValueError(f"Channel mismatch: z channels={cz}, x channels={cx}")
        if hz > hx or wz > wx:
            raise ValueError(
                f"Template feature map must not be larger than search feature map: "
                f"z={tuple(z.shape)}, x={tuple(x.shape)}"
            )

        batch_size, channels = bz, cz

        # Grouped convolution trick:
        # reshape x to [1, B*C, Hx, Wx]
        # reshape z to [B*C, 1, Hz, Wz]
        # groups = B*C
        # => each channel of each sample is correlated independently
        x_reshaped = x.reshape(1, batch_size * channels, hx, wx)
        z_reshaped = z.reshape(batch_size * channels, 1, hz, wz)

        response = F.conv2d(
            input=x_reshaped,
            weight=z_reshaped,
            bias=None,
            stride=1,
            padding=0,
            groups=batch_size * channels,
        )

        hr, wr = response.shape[-2:]
        response = response.reshape(batch_size, channels, hr, wr)
        return response

    def forward_with_output(self, z: torch.Tensor, x: torch.Tensor) -> CorrelationOutput:
        return CorrelationOutput(response=self.forward(z, x))


# ─── Siamese Head ─────────────────────────────────────────────────────────────

class SiameseHead(nn.Module):
    def __init__(self, in_channels: int = 96):
        super().__init__()

        # Classification branch
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        # BBox regression branch
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        cls_logits = self.cls_head(x)   # [B, 1, 5, 5]
        bbox_pred = self.reg_head(x)    # [B, 4, 5, 5]

        return {
            "cls_logits": cls_logits,
            "bbox_pred": bbox_pred
        }


# ─── Siamese Tracker ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SiameseTrackerOutput:
    template_features: torch.Tensor
    search_features: torch.Tensor
    response_map: torch.Tensor
    cls_logits: torch.Tensor
    bbox_pred: torch.Tensor


class SiameseTracker(nn.Module):
    """
    Siamese single-object tracker.

    Pipeline:
        template -> shared backbone -> z_feat
        search   -> shared backbone -> x_feat
        z_feat + x_feat -> depthwise cross-correlation -> response map
        response map -> head -> classification + bbox regression
    """

    def __init__(
        self,
        backbone_variant: str = "small",
        pretrained_backbone: bool = True,
        feature_channels: int = 96,
        freeze_backbone: bool = True,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = MobileNetV3Backbone(
            variant=backbone_variant,
            pretrained=pretrained_backbone,
            out_channels=feature_channels,
            trainable=not freeze_backbone,
            normalize_input=normalize_input,
        )
        self.correlation = DepthwiseCrossCorrelation()
        self.head = SiameseHead(in_channels=feature_channels)

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> dict[str, torch.Tensor]:
        if template.ndim != 4 or search.ndim != 4:
            raise ValueError(
                f"Expected template/search to be 4D [B, C, H, W], "
                f"got template={tuple(template.shape)}, search={tuple(search.shape)}"
            )

        if template.size(0) != search.size(0):
            raise ValueError(
                f"Batch size mismatch: template batch={template.size(0)}, search batch={search.size(0)}"
            )

        template_features = self.backbone(template)
        search_features = self.backbone(search)

        response_map = self.correlation(template_features, search_features)
        head_outputs = self.head(response_map)

        return {
            "template_features": template_features,
            "search_features": search_features,
            "response_map": response_map,
            "cls_logits": head_outputs["cls_logits"],
            "bbox_pred": head_outputs["bbox_pred"],
        }

    def forward_with_output(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
    ) -> SiameseTrackerOutput:
        outputs = self.forward(template, search)
        return SiameseTrackerOutput(
            template_features=outputs["template_features"],
            search_features=outputs["search_features"],
            response_map=outputs["response_map"],
            cls_logits=outputs["cls_logits"],
            bbox_pred=outputs["bbox_pred"],
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseTracker(
        backbone_variant="small",
        pretrained_backbone=True,
        feature_channels=96,
        freeze_backbone=True,
        normalize_input=True,
    ).to(device)

    template = torch.randn(2, 3, 127, 127, device=device)
    search = torch.randn(2, 3, 255, 255, device=device)

    outputs = model(template, search)

    print("template input:", template.shape)
    print("search input:", search.shape)
    print("template features:", outputs["template_features"].shape)
    print("search features:", outputs["search_features"].shape)
    print("response map:", outputs["response_map"].shape)
    print("cls logits:", outputs["cls_logits"].shape)
    print("bbox pred:", outputs["bbox_pred"].shape)
