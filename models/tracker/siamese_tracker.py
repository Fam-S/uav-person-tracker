from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.backbone.mobilenetv3_tracker import MobileNetV3Backbone
from models.tracker.correlation import DepthwiseCrossCorrelation
from models.tracker.head import SiameseHead


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