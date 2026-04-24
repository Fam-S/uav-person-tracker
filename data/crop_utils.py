from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import Tensor


def xywh_to_center(box_xywh: np.ndarray | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return x + (w / 2.0), y + (h / 2.0), w, h


def compute_crop_size(
    box_xywh: np.ndarray | tuple[float, float, float, float],
    context_amount: float,
    area_scale: float = 1.0,
) -> float:
    _, _, box_w, box_h = xywh_to_center(box_xywh)
    context = context_amount * (box_w + box_h)
    crop_size = math.sqrt((box_w + context) * (box_h + context))
    return max(2.0, crop_size * float(area_scale))


def crop_and_resize(
    frame: np.ndarray,
    box_xywh: np.ndarray | tuple[float, float, float, float],
    out_size: int,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    center_x, center_y, _, _ = xywh_to_center(box_xywh)
    if center_override is not None:
        center_x, center_y = center_override

    crop_size = compute_crop_size(box_xywh, context_amount=context_amount, area_scale=area_scale)

    x1 = center_x - (crop_size / 2.0)
    y1 = center_y - (crop_size / 2.0)
    x2 = center_x + (crop_size / 2.0)
    y2 = center_y + (crop_size / 2.0)

    left_pad = max(0, int(math.ceil(-x1)))
    top_pad = max(0, int(math.ceil(-y1)))
    right_pad = max(0, int(math.ceil(x2 - frame_w)))
    bottom_pad = max(0, int(math.ceil(y2 - frame_h)))

    if left_pad or top_pad or right_pad or bottom_pad:
        frame = cv2.copyMakeBorder(
            frame,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            borderType=cv2.BORDER_REPLICATE,
        )

    x1 += left_pad
    x2 += left_pad
    y1 += top_pad
    y2 += top_pad

    x1_i = int(round(x1))
    y1_i = int(round(y1))
    x2_i = int(round(x2))
    y2_i = int(round(y2))

    crop = frame[y1_i:y2_i, x1_i:x2_i]
    if crop.size == 0:
        raise ValueError("Crop produced an empty patch.")
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def project_box_to_crop(
    search_box_xywh: np.ndarray | tuple[float, float, float, float],
    reference_box_xywh: np.ndarray | tuple[float, float, float, float],
    out_size: int,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
) -> np.ndarray:
    center_x, center_y, _, _ = xywh_to_center(reference_box_xywh)
    if center_override is not None:
        center_x, center_y = center_override

    crop_size = compute_crop_size(reference_box_xywh, context_amount=context_amount, area_scale=area_scale)
    scale = float(out_size) / crop_size

    x, y, w, h = [float(v) for v in search_box_xywh]
    crop_x1 = center_x - (crop_size / 2.0)
    crop_y1 = center_y - (crop_size / 2.0)

    projected_x1 = (x - crop_x1) * scale
    projected_y1 = (y - crop_y1) * scale
    projected_x2 = (x + w - crop_x1) * scale
    projected_y2 = (y + h - crop_y1) * scale

    clipped_x1 = float(np.clip(projected_x1, a_min=0.0, a_max=float(out_size)))
    clipped_y1 = float(np.clip(projected_y1, a_min=0.0, a_max=float(out_size)))
    clipped_x2 = float(np.clip(projected_x2, a_min=0.0, a_max=float(out_size)))
    clipped_y2 = float(np.clip(projected_y2, a_min=0.0, a_max=float(out_size)))

    projected = np.asarray(
        [
            clipped_x1,
            clipped_y1,
            max(0.0, clipped_x2 - clipped_x1),
            max(0.0, clipped_y2 - clipped_y1),
        ],
        dtype=np.float32,
    )
    return projected


def project_box_to_crop_center_norm(
    search_box_xywh: np.ndarray | tuple[float, float, float, float],
    reference_box_xywh: np.ndarray | tuple[float, float, float, float],
    out_size: int,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
) -> np.ndarray:
    crop_box = project_box_to_crop(
        search_box_xywh=search_box_xywh,
        reference_box_xywh=reference_box_xywh,
        out_size=out_size,
        context_amount=context_amount,
        center_override=center_override,
        area_scale=area_scale,
    )
    x, y, w, h = [float(v) for v in crop_box]
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    return np.asarray([center_x, center_y, w, h], dtype=np.float32) / float(out_size)


def project_box_from_crop(
    crop_box_xywh: np.ndarray | tuple[float, float, float, float],
    reference_box_xywh: np.ndarray | tuple[float, float, float, float],
    out_size: int,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    center_x, center_y, _, _ = xywh_to_center(reference_box_xywh)
    if center_override is not None:
        center_x, center_y = center_override

    crop_size = compute_crop_size(reference_box_xywh, context_amount=context_amount, area_scale=area_scale)
    scale = crop_size / float(out_size)
    crop_x1 = center_x - (crop_size / 2.0)
    crop_y1 = center_y - (crop_size / 2.0)

    x, y, w, h = [float(v) for v in crop_box_xywh]
    frame_x = crop_x1 + x * scale
    frame_y = crop_y1 + y * scale
    frame_w = w * scale
    frame_h = h * scale

    if frame_shape is not None:
        image_h, image_w = frame_shape
        frame_x = max(0.0, min(frame_x, image_w - 1.0))
        frame_y = max(0.0, min(frame_y, image_h - 1.0))
        frame_w = max(1.0, min(frame_w, image_w - frame_x))
        frame_h = max(1.0, min(frame_h, image_h - frame_y))

    return int(round(frame_x)), int(round(frame_y)), int(round(frame_w)), int(round(frame_h))


def project_box_from_crop_center_norm(
    crop_box_cxcywh: np.ndarray | tuple[float, float, float, float],
    reference_box_xywh: np.ndarray | tuple[float, float, float, float],
    out_size: int,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    cx, cy, w, h = [float(v) for v in crop_box_cxcywh]
    crop_box_xywh = np.asarray(
        [
            (cx * out_size) - ((w * out_size) / 2.0),
            (cy * out_size) - ((h * out_size) / 2.0),
            w * out_size,
            h * out_size,
        ],
        dtype=np.float32,
    )
    return project_box_from_crop(
        crop_box_xywh=crop_box_xywh,
        reference_box_xywh=reference_box_xywh,
        out_size=out_size,
        context_amount=context_amount,
        center_override=center_override,
        area_scale=area_scale,
        frame_shape=frame_shape,
    )


def frame_to_tensor(frame_bgr: np.ndarray) -> Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().float()
    return tensor / 255.0
