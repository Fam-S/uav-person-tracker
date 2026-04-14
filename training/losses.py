from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SiameseTargets:
    cls_target: torch.Tensor
    reg_target: torch.Tensor
    reg_mask: torch.Tensor
    positive_indices: torch.Tensor


@dataclass(frozen=True)
class SiameseLossOutput:
    total_loss: torch.Tensor
    cls_loss: torch.Tensor
    reg_loss: torch.Tensor
    cls_target: torch.Tensor
    reg_target: torch.Tensor
    reg_mask: torch.Tensor
    positive_indices: torch.Tensor


def _clamp_indices(value: torch.Tensor, max_value: int) -> torch.Tensor:
    return value.clamp(min=0, max=max_value - 1)


def normalize_bbox_xywh(search_bbox: torch.Tensor, search_size: int) -> torch.Tensor:
    if search_bbox.ndim != 2 or search_bbox.size(1) != 4:
        raise ValueError(f"Expected search_bbox shape [B, 4], got {tuple(search_bbox.shape)}")
    return search_bbox / float(search_size)


def build_targets(
    search_bbox: torch.Tensor,
    cls_logits: torch.Tensor,
    search_size: int = 255,
) -> SiameseTargets:
    if search_bbox.ndim != 2 or search_bbox.size(1) != 4:
        raise ValueError(f"Expected search_bbox shape [B, 4], got {tuple(search_bbox.shape)}")

    if cls_logits.ndim != 4 or cls_logits.size(1) != 1:
        raise ValueError(f"Expected cls_logits shape [B, 1, H, W], got {tuple(cls_logits.shape)}")

    device = cls_logits.device
    dtype = cls_logits.dtype

    batch_size, _, grid_h, grid_w = cls_logits.shape

    if search_bbox.size(0) != batch_size:
        raise ValueError(
            f"Batch mismatch between search_bbox and cls_logits: "
            f"{search_bbox.size(0)} vs {batch_size}"
        )

    cls_target = torch.zeros((batch_size, 1, grid_h, grid_w), device=device, dtype=dtype)
    reg_target = torch.zeros((batch_size, 4, grid_h, grid_w), device=device, dtype=dtype)
    reg_mask = torch.zeros((batch_size, 1, grid_h, grid_w), device=device, dtype=dtype)

    x, y, w, h = search_bbox.unbind(dim=1)
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)

    gx = torch.floor(center_x / float(search_size) * grid_w).long()
    gy = torch.floor(center_y / float(search_size) * grid_h).long()

    gx = _clamp_indices(gx, grid_w)
    gy = _clamp_indices(gy, grid_h)

    positive_indices = torch.stack([gy, gx], dim=1)
    normalized_bbox = normalize_bbox_xywh(search_bbox, search_size=search_size).to(device=device, dtype=dtype)

    for b in range(batch_size):
        cls_target[b, 0, gy[b], gx[b]] = 1.0
        reg_mask[b, 0, gy[b], gx[b]] = 1.0

        reg_target[b, 0, :, :] = normalized_bbox[b, 0]
        reg_target[b, 1, :, :] = normalized_bbox[b, 1]
        reg_target[b, 2, :, :] = normalized_bbox[b, 2]
        reg_target[b, 3, :, :] = normalized_bbox[b, 3]

    return SiameseTargets(
        cls_target=cls_target,
        reg_target=reg_target,
        reg_mask=reg_mask,
        positive_indices=positive_indices,
    )


class SiameseLoss(nn.Module):
    """
    Baseline loss:
    - BCEWithLogits for classification
    - SmoothL1 for bbox regression on positive cell only
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        search_size: int = 255,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.search_size = int(search_size)
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.SmoothL1Loss(reduction="none", beta=float(smooth_l1_beta))

    def forward(
        self,
        cls_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        search_bbox: torch.Tensor,
    ) -> SiameseLossOutput:
        if bbox_pred.ndim != 4 or bbox_pred.size(1) != 4:
            raise ValueError(f"Expected bbox_pred shape [B, 4, H, W], got {tuple(bbox_pred.shape)}")

        if cls_logits.size(0) != bbox_pred.size(0):
            raise ValueError(
                f"Batch mismatch between cls_logits and bbox_pred: "
                f"{cls_logits.size(0)} vs {bbox_pred.size(0)}"
            )

        targets = build_targets(
            search_bbox=search_bbox,
            cls_logits=cls_logits,
            search_size=self.search_size,
        )

        cls_loss = self.cls_loss_fn(cls_logits, targets.cls_target)

        reg_loss_per_element = self.reg_loss_fn(bbox_pred, targets.reg_target)
        reg_mask_expanded = targets.reg_mask.expand_as(bbox_pred)

        positive_count = reg_mask_expanded.sum().clamp_min(1.0)
        reg_loss = (reg_loss_per_element * reg_mask_expanded).sum() / positive_count

        total_loss = (self.cls_weight * cls_loss) + (self.reg_weight * reg_loss)

        return SiameseLossOutput(
            total_loss=total_loss,
            cls_loss=cls_loss,
            reg_loss=reg_loss,
            cls_target=targets.cls_target,
            reg_target=targets.reg_target,
            reg_mask=targets.reg_mask,
            positive_indices=targets.positive_indices,
        )


if __name__ == "__main__":
    batch_size = 2
    grid_h, grid_w = 5, 5

    cls_logits = torch.randn(batch_size, 1, grid_h, grid_w)
    bbox_pred = torch.randn(batch_size, 4, grid_h, grid_w)

    search_bbox = torch.tensor(
        [
            [73.0, 75.0, 38.0, 107.0],
            [90.0, 60.0, 40.0, 80.0],
        ],
        dtype=torch.float32,
    )

    criterion = SiameseLoss(
        cls_weight=1.0,
        reg_weight=1.0,
        search_size=255,
        smooth_l1_beta=1.0,
    )

    loss_out = criterion(
        cls_logits=cls_logits,
        bbox_pred=bbox_pred,
        search_bbox=search_bbox,
    )

    print("cls_logits:", cls_logits.shape)
    print("bbox_pred:", bbox_pred.shape)
    print("cls_target:", loss_out.cls_target.shape)
    print("reg_target:", loss_out.reg_target.shape)
    print("reg_mask:", loss_out.reg_mask.shape)
    print("positive_indices:", loss_out.positive_indices)
    print("cls_loss:", float(loss_out.cls_loss))
    print("reg_loss:", float(loss_out.reg_loss))
    print("total_loss:", float(loss_out.total_loss))