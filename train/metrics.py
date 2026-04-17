from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PredictionSelection:
    pred_bbox: torch.Tensor
    pred_scores: torch.Tensor
    pred_indices: torch.Tensor


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.size(-1) != 4:
        raise ValueError(f"Expected last dimension = 4, got {boxes.shape}")

    x, y, w, h = boxes.unbind(dim=-1)
    x2 = x + w
    y2 = y + h
    return torch.stack([x, y, x2, y2], dim=-1)


def compute_iou_xywh(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.ndim != 2 or target_boxes.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors [B, 4], got pred={tuple(pred_boxes.shape)}, target={tuple(target_boxes.shape)}"
        )
    if pred_boxes.shape != target_boxes.shape:
        raise ValueError(
            f"Shape mismatch: pred={tuple(pred_boxes.shape)}, target={tuple(target_boxes.shape)}"
        )
    if pred_boxes.size(1) != 4:
        raise ValueError(f"Expected shape [B, 4], got pred={tuple(pred_boxes.shape)}")

    pred_xyxy = xywh_to_xyxy(pred_boxes)
    target_xyxy = xywh_to_xyxy(target_boxes)

    pred_x1, pred_y1, pred_x2, pred_y2 = pred_xyxy.unbind(dim=1)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_xyxy.unbind(dim=1)

    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_x2 - pred_x1).clamp(min=0.0) * (pred_y2 - pred_y1).clamp(min=0.0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(min=0.0) * (tgt_y2 - tgt_y1).clamp(min=0.0)

    union = pred_area + tgt_area - inter_area
    iou = inter_area / union.clamp(min=1e-6)
    return iou


def select_top_prediction(
    cls_logits: torch.Tensor,
    bbox_pred: torch.Tensor,
    search_size: int = 255,
) -> PredictionSelection:
    if cls_logits.ndim != 4 or cls_logits.size(1) != 1:
        raise ValueError(f"Expected cls_logits shape [B, 1, H, W], got {tuple(cls_logits.shape)}")
    if bbox_pred.ndim != 4 or bbox_pred.size(1) != 4:
        raise ValueError(f"Expected bbox_pred shape [B, 4, H, W], got {tuple(bbox_pred.shape)}")
    if cls_logits.size(0) != bbox_pred.size(0):
        raise ValueError(
            f"Batch mismatch: cls_logits batch={cls_logits.size(0)}, bbox_pred batch={bbox_pred.size(0)}"
        )

    batch_size, _, cls_h, cls_w = cls_logits.shape
    _, _, reg_h, reg_w = bbox_pred.shape

    scores = torch.sigmoid(cls_logits).view(batch_size, -1)
    top_scores, top_indices = torch.max(scores, dim=1)

    gy_cls = torch.div(top_indices, cls_w, rounding_mode="floor")
    gx_cls = top_indices % cls_w
    pred_indices = torch.stack([gy_cls, gx_cls], dim=1)

    stride_h = cls_h // reg_h
    stride_w = cls_w // reg_w

    gy_reg = (gy_cls / stride_h).long().clamp(max=reg_h - 1)
    gx_reg = (gx_cls / stride_w).long().clamp(max=reg_w - 1)

    reg_stride = float(search_size) / float(reg_w)

    pred_boxes = []
    for b in range(batch_size):
        tx = bbox_pred[b, 0, gy_reg[b], gx_reg[b]]
        ty = bbox_pred[b, 1, gy_reg[b], gx_reg[b]]
        tw = bbox_pred[b, 2, gy_reg[b], gx_reg[b]]
        th = bbox_pred[b, 3, gy_reg[b], gx_reg[b]]

        anchor_cx = (gx_reg[b].float() + 0.5) * reg_stride
        anchor_cy = (gy_reg[b].float() + 0.5) * reg_stride

        pred_cx = anchor_cx + tx * reg_stride
        pred_cy = anchor_cy + ty * reg_stride

        pred_w = torch.clamp(torch.exp(tw), max=10.0) * reg_stride
        pred_h = torch.clamp(torch.exp(th), max=10.0) * reg_stride

        pred_x = pred_cx - (pred_w / 2.0)
        pred_y = pred_cy - (pred_h / 2.0)

        pred_boxes.append(torch.stack([pred_x, pred_y, pred_w, pred_h]))

    pred_bbox = torch.stack(pred_boxes, dim=0)

    return PredictionSelection(
        pred_bbox=pred_bbox,
        pred_scores=top_scores,
        pred_indices=pred_indices,
    )


def compute_batch_metrics(
    cls_logits: torch.Tensor,
    bbox_pred: torch.Tensor,
    search_bbox: torch.Tensor,
    search_size: int = 255,
) -> dict[str, torch.Tensor]:
    selection = select_top_prediction(
        cls_logits=cls_logits,
        bbox_pred=bbox_pred,
        search_size=search_size,
    )

    ious = compute_iou_xywh(
        pred_boxes=selection.pred_bbox,
        target_boxes=search_bbox,
    )

    return {
        "mean_iou": ious.mean(),
        "mean_score": selection.pred_scores.mean(),
        "pred_bbox": selection.pred_bbox,
        "pred_scores": selection.pred_scores,
        "pred_indices": selection.pred_indices,
        "ious": ious,
    }
