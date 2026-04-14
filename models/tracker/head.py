import torch
import torch.nn as nn


class SiameseHead(nn.Module):
    """
    Head for Siamese tracker:
    - Classification branch
    - Bounding box regression branch
    """

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
        cls_logits = self.cls_head(x)   # [B, 1, H, W]
        bbox_pred = self.reg_head(x)    # [B, 4, H, W]

        return {
            "cls_logits": cls_logits,
            "bbox_pred": bbox_pred
        }


if __name__ == "__main__":
    x = torch.randn(2, 96, 5, 5)

    head = SiameseHead(in_channels=96)
    out = head(x)

    print("input:", x.shape)
    print("cls:", out["cls_logits"].shape)
    print("bbox:", out["bbox_pred"].shape)