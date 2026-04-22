from __future__ import annotations

import torch


def decode_predicted_boxes(bbox_pred: torch.Tensor, search_size: int = 255) -> torch.Tensor:
    if bbox_pred.ndim != 2 or bbox_pred.size(1) != 4:
        raise ValueError(f"Expected bbox_pred shape [B, 4], got {tuple(bbox_pred.shape)}")
    clipped = bbox_pred.clone()
    clipped[:, :2] = clipped[:, :2].clamp(min=0.0, max=float(search_size))
    clipped[:, 2:] = clipped[:, 2:].clamp(min=1.0, max=float(search_size))
    return clipped


def _box_iou_xywh(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    ax1, ay1, aw, ah = box_a.unbind(dim=-1)
    bx1, by1, bw, bh = box_b.unbind(dim=-1)
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)
    inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
    inter_area = inter_w * inter_h

    area_a = aw.clamp_min(0.0) * ah.clamp_min(0.0)
    area_b = bw.clamp_min(0.0) * bh.clamp_min(0.0)
    union = area_a + area_b - inter_area
    return inter_area / union.clamp_min(1e-6)


def compute_batch_metrics(
    bbox_pred: torch.Tensor,
    search_bbox: torch.Tensor,
    search_size: int = 255,
) -> dict[str, torch.Tensor]:
    pred_bbox = decode_predicted_boxes(bbox_pred=bbox_pred, search_size=search_size)
    return {"pred_bbox": pred_bbox, "ious": _box_iou_xywh(pred_bbox, search_bbox)}
