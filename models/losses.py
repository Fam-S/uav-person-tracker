from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cls_loss(pred: torch.Tensor, label: torch.Tensor, select: torch.Tensor) -> torch.Tensor:
    if select.numel() == 0:
        return pred.new_zeros(())
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select).long()
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = torch.nonzero(label == 1, as_tuple=False).squeeze(1).to(pred.device)
    neg = torch.nonzero(label == 0, as_tuple=False).squeeze(1).to(pred.device)
    return get_cls_loss(pred, label, pos) * 0.5 + get_cls_loss(pred, label, neg) * 0.5


def shaloss(pred: torch.Tensor, label: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = torch.abs(pred - label)
    quadratic = loss < 0.04
    loss = torch.where(quadratic, 25 * loss.pow(2), loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def weight_l1_loss(pred_loc: torch.Tensor, label_loc: torch.Tensor, loss_weight: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = pred_loc.size()
    pred_loc = pred_loc.view(batch, 4, -1, height, width)
    diff = (pred_loc - label_loc).abs()
    diff = torch.where(diff < 0.001, 1000 * diff.pow(2), diff)
    diff = diff.sum(dim=1).view(batch, -1, height, width)
    return (diff * loss_weight).sum().div(batch)


class IOULoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        pred_left = pred[:, :, 0]
        pred_top = pred[:, :, 1]
        pred_right = pred[:, :, 2]
        pred_bottom = pred[:, :, 3]
        target_left = target[:, :, 0]
        target_top = target[:, :, 1]
        target_right = target[:, :, 2]
        target_bottom = target[:, :, 3]

        target_area = (target_right - target_left) * (target_bottom - target_top)
        pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)
        w_intersect = (torch.min(pred_right, target_right) - torch.max(pred_left, target_left)).clamp(min=0)
        h_intersect = (torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)).clamp(min=0)
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect / (area_union + 1e-6)).clamp(min=0) + 1e-6
        losses = -(1 - ious) * (1.5 - ious) * torch.log(ious)
        if weight is None:
            return losses.mean()
        weight = weight.view(losses.size())
        if weight.sum() > 0:
            return (losses * weight).sum() / (weight.sum() + 1e-6)
        return (losses * weight).sum()


class SiamAPNLoss(nn.Module):
    """Compatibility shell; full SiamAPN++ losses are computed inside the model."""

    def forward(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return outputs
