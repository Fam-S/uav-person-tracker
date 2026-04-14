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


def denormalize_bbox_xywh(boxes: torch.Tensor, search_size: int) -> torch.Tensor:
    if boxes.size(-1) != 4:
        raise ValueError(f"Expected last dimension = 4, got {boxes.shape}")
    return boxes * float(search_size)


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
    if cls_logits.shape[-2:] != bbox_pred.shape[-2:]:
        raise ValueError(
            f"Spatial mismatch: cls_logits HW={cls_logits.shape[-2:]}, bbox_pred HW={bbox_pred.shape[-2:]}"
        )

    batch_size, _, grid_h, grid_w = cls_logits.shape

    scores = torch.sigmoid(cls_logits).view(batch_size, -1)
    top_scores, top_indices = torch.max(scores, dim=1)

    gy = torch.div(top_indices, grid_w, rounding_mode="floor")
    gx = top_indices % grid_w
    pred_indices = torch.stack([gy, gx], dim=1)

    pred_boxes = []
    for b in range(batch_size):
        pred_boxes.append(bbox_pred[b, :, gy[b], gx[b]])

    pred_bbox = torch.stack(pred_boxes, dim=0)
    pred_bbox = denormalize_bbox_xywh(pred_bbox, search_size=search_size)

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

    metrics = compute_batch_metrics(
        cls_logits=cls_logits,
        bbox_pred=bbox_pred,
        search_bbox=search_bbox,
        search_size=255,
    )

    print("pred_bbox:", metrics["pred_bbox"].shape)
    print("pred_scores:", metrics["pred_scores"].shape)
    print("pred_indices:", metrics["pred_indices"])
    print("ious:", metrics["ious"])
    print("mean_iou:", float(metrics["mean_iou"]))
    print("mean_score:", float(metrics["mean_score"]))