import torch
import torch.nn as nn


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