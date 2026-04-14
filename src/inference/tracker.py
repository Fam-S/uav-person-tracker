from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math

import cv2
import numpy as np

from src.inference.predictor import Predictor, PredictionResult


@dataclass(frozen=True)
class TrackerResult:
    bbox: Tuple[float, float, float, float]
    score: float
    state: str
    search_bbox: Tuple[int, int, int, int]
    grid_pos: Tuple[int, int]


def crop_square_patch_cv2(image, bbox_xywh, output_size, context_amount=0.5, scale_multiplier=1.0):
    """EXACT SAME CROP LOGIC AS TRAINING DATASET"""
    x, y, w, h = bbox_xywh
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    context = context_amount * (w + h)
    base_crop_side = math.sqrt((w + context) * (h + context)) * scale_multiplier
    crop_side = max(base_crop_side, 1.0)
    
    half_side = crop_side / 2.0
    left = center_x - half_side
    top = center_y - half_side
    right = center_x + half_side
    bottom = center_y + half_side
    
    H, W = image.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(W, right), min(H, bottom)
    
    crop = image[int(top):int(bottom), int(left):int(right)]
    if crop.size == 0: raise ValueError("Crop empty")
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


class SiameseTrackerInference:
    def __init__(self, predictor, template_size=127, search_size=255, context_amount=0.5, tracking_threshold=0.6, uncertain_threshold=0.3):
        self.predictor = predictor
        self.template_size = int(template_size)
        self.search_size = int(search_size)
        self.context_amount = float(context_amount)
        self.tracking_threshold = float(tracking_threshold)
        self.uncertain_threshold = float(uncertain_threshold)

        self.template_image = None
        self.current_bbox = None
        self.last_score = 0.0
        self.state = "Lost"

    def initialize(self, frame, bbox):
        self.current_bbox = tuple(map(float, bbox))
        # Use EXACT training crop for template
        self.template_image = crop_square_patch_cv2(frame, self.current_bbox, self.template_size, self.context_amount, 1.0)
        self.last_score = 1.0
        self.state = "Tracking"
        return TrackerResult(bbox=self.current_bbox, score=self.last_score, state=self.state, search_bbox=tuple(map(int, self.current_bbox)), grid_pos=(0, 0))

    def track(self, frame):
        # Use EXACT training crop for search
        search_crop = crop_square_patch_cv2(frame, self.current_bbox, self.search_size, self.context_amount, 2.0)
        
        pred = self.predictor.predict(self.template_image, search_crop)
        new_bbox = self._decode_prediction_to_frame_bbox(pred)
        
        self.current_bbox = new_bbox
        self.last_score = pred.score
        self.state = self._score_to_state(pred.score)
        
        return TrackerResult(bbox=new_bbox, score=pred.score, state=self.state, search_bbox=(0,0,0,0), grid_pos=(pred.grid_x, pred.grid_y))

    def _score_to_state(self, score):
        if score >= self.tracking_threshold: return "Tracking"
        if score >= self.uncertain_threshold: return "Uncertain"
        return "Lost"

    def _decode_prediction_to_frame_bbox(self, pred):
        tx, ty, tw, th = [float(v) for v in pred.bbox_raw]
        
        # 5x5 Grid Math
        reg_stride = float(self.search_size) / 5.0
        anchor_cx = (pred.grid_x + 0.5) * reg_stride
        anchor_cy = (pred.grid_y + 0.5) * reg_stride
        
        # Use tx, ty to find exact center in 255x255 space
        pred_cx = anchor_cx + tx * reg_stride
        pred_cy = anchor_cy + ty * reg_stride
        
        # Convert center to top-left corner
        pred_x_255 = pred_cx - (self.search_size / 2.0)
        pred_y_255 = pred_cy - (self.search_size / 2.0)
        
        # Scale from 255x255 crop back to original frame coordinates
        # (Since we used crop_square_patch_cv2, the crop covers the target area perfectly)
        frame_x = self.current_bbox[0] + pred_x_255
        frame_y = self.current_bbox[1] + pred_y_255
        
        # FREEZE SIZE: Keep the exact width and height from initialization
        frame_w = self.current_bbox[2]
        frame_h = self.current_bbox[3]

        return (frame_x, frame_y, frame_w, frame_h)