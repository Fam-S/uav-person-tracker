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


def build_targets(
    search_bbox: torch.Tensor,
    cls_logits: torch.Tensor,
    search_size: int = 255,
    reg_grid_h: int = 5,
    reg_grid_w: int = 5,
    gaussian_penalty_k: float = 0.1,
) -> SiameseTargets:
    if search_bbox.ndim != 2 or search_bbox.size(1) != 4:
        raise ValueError(f"Expected search_bbox shape [B, 4], got {tuple(search_bbox.shape)}")

    if cls_logits.ndim != 4 or cls_logits.size(1) != 1:
        raise ValueError(f"Expected cls_logits shape [B, 1, H, W], got {tuple(cls_logits.shape)}")

    device = cls_logits.device
    dtype = cls_logits.dtype

    batch_size, _, cls_h, cls_w = cls_logits.shape

    if search_bbox.size(0) != batch_size:
        raise ValueError(
            f"Batch mismatch between search_bbox and cls_logits: "
            f"{search_bbox.size(0)} vs {batch_size}"
        )

    # cls_target is built on the HIGH resolution grid (e.g., 25x25)
    cls_target = torch.zeros((batch_size, 1, cls_h, cls_w), device=device, dtype=dtype)

    # reg_target is built on the LOW resolution grid (e.g., 5x5)
    reg_target = torch.zeros((batch_size, 4, reg_grid_h, reg_grid_w), device=device, dtype=dtype)
    reg_mask = torch.zeros((batch_size, 1, reg_grid_h, reg_grid_w), device=device, dtype=dtype)

    x, y, w, h = search_bbox.unbind(dim=1)
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)

    # 1. Map to REG grid (5x5) for offsets
    reg_stride = float(search_size) / float(reg_grid_w)
    gx_reg = torch.floor(center_x / reg_stride).long()
    gy_reg = torch.floor(center_y / reg_stride).long()
    gx_reg = _clamp_indices(gx_reg, reg_grid_w)
    gy_reg = _clamp_indices(gy_reg, reg_grid_h)

    positive_indices = torch.stack([gy_reg, gx_reg], dim=1)

    # 2. Map to CLS grid (25x25) for gaussian
    cls_stride = float(search_size) / float(cls_w)
    gx_cls = torch.floor(center_x / cls_stride).long()
    gy_cls = torch.floor(center_y / cls_stride).long()
    gx_cls = _clamp_indices(gx_cls, cls_w)
    gy_cls = _clamp_indices(gy_cls, cls_h)

    # Build 2D Gaussian Heatmap on CLS grid
    y_grid = torch.arange(0, cls_h, device=device, dtype=dtype)
    x_grid = torch.arange(0, cls_w, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')

    for b in range(batch_size):
        target_size = torch.sqrt(w[b] * h[b])
        sigma = target_size / (cls_w * gaussian_penalty_k)
        sigma = torch.clamp(sigma, min=1.0)

        dist = (x_grid - gx_cls[b].float())**2 + (y_grid - gy_cls[b].float())**2
        gaussian_map = torch.exp(-dist / (2 * sigma**2))

        cls_target[b, 0, :, :] = gaussian_map

        # Build Offsets on REG grid
        reg_mask[b, 0, gy_reg[b], gx_reg[b]] = 1.0

        anchor_cx = (gx_reg[b].float() + 0.5) * reg_stride
        anchor_cy = (gy_reg[b].float() + 0.5) * reg_stride

        # Calculate Offset and Log Scale
        tx = (center_x[b] - anchor_cx) / reg_stride
        ty = (center_y[b] - anchor_cy) / reg_stride

        # Clamping w/h before log to prevent NaN if a box is extremely small
        w_clamped = torch.clamp(w[b], min=1e-4)
        h_clamped = torch.clamp(h[b], min=1e-4)
        tw = torch.log(w_clamped / reg_stride)
        th = torch.log(h_clamped / reg_stride)

        reg_target[b, 0, gy_reg[b], gx_reg[b]] = tx
        reg_target[b, 1, gy_reg[b], gx_reg[b]] = ty
        reg_target[b, 2, gy_reg[b], gx_reg[b]] = tw
        reg_target[b, 3, gy_reg[b], gx_reg[b]] = th

    return SiameseTargets(
        cls_target=cls_target,
        reg_target=reg_target,
        reg_mask=reg_mask,
        positive_indices=positive_indices,
    )


class SiameseLoss(nn.Module):
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

        # Extract the spatial dimensions of the regression map to pass to build_targets
        reg_grid_h, reg_grid_w = bbox_pred.shape[-2:]

        targets = build_targets(
            search_bbox=search_bbox,
            cls_logits=cls_logits,
            search_size=self.search_size,
            reg_grid_h=reg_grid_h,
            reg_grid_w=reg_grid_w,
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
