from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SiameseLossOutput:
    total_loss: torch.Tensor
    reg_loss: torch.Tensor
class SiameseLoss(nn.Module):
    def __init__(
        self,
        search_size: int = 255,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.search_size = int(search_size)
        self.reg_loss_fn = nn.SmoothL1Loss(beta=float(smooth_l1_beta))

    def forward(
        self,
        bbox_pred: torch.Tensor,
        search_bbox: torch.Tensor,
    ) -> SiameseLossOutput:
        if bbox_pred.ndim != 2 or bbox_pred.size(1) != 4:
            raise ValueError(f"Expected bbox_pred shape [B, 4], got {tuple(bbox_pred.shape)}")
        if search_bbox.ndim != 2 or search_bbox.size(1) != 4:
            raise ValueError(f"Expected search_bbox shape [B, 4], got {tuple(search_bbox.shape)}")
        if bbox_pred.size(0) != search_bbox.size(0):
            raise ValueError(
                f"Batch mismatch between bbox_pred and search_bbox: "
                f"{bbox_pred.size(0)} vs {search_bbox.size(0)}"
            )

        reg_loss = self.reg_loss_fn(bbox_pred, search_bbox)
        total_loss = reg_loss

        return SiameseLossOutput(
            total_loss=total_loss,
            reg_loss=reg_loss,
        )


class SiamAPNLoss(SiameseLoss):
    """Alias kept for the new minimal SiamAPN-style training loop."""
