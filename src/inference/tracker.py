from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from src.inference.predictor import Predictor, PredictionResult


@dataclass(frozen=True)
class TrackerResult:
    bbox: Tuple[float, float, float, float]   # x, y, w, h in full-frame coords
    score: float
    state: str
    search_bbox: Tuple[int, int, int, int]    # x, y, w, h of search crop in frame coords
    grid_pos: Tuple[int, int]


class SiameseTrackerInference:
    """
    Runtime Siamese tracker.

    Notes:
    - Input bbox format: (x, y, w, h)
    - Predictor returns the best cls score and bbox_raw
    - bbox_raw is interpreted as normalized [x, y, w, h] inside the resized search crop
    - The normalized bbox is converted back to full-frame coordinates through the search region
    """

    def __init__(
        self,
        predictor: Predictor,
        template_size: int = 127,
        search_size: int = 255,
        search_scale: float = 2.0,
        tracking_threshold: float = 0.6,
        uncertain_threshold: float = 0.3,
        min_box_size: float = 10.0,
    ) -> None:
        self.predictor = predictor
        self.template_size = int(template_size)
        self.search_size = int(search_size)
        self.search_scale = float(search_scale)

        self.tracking_threshold = float(tracking_threshold)
        self.uncertain_threshold = float(uncertain_threshold)
        self.min_box_size = float(min_box_size)

        self.template_image: np.ndarray | None = None
        self.current_bbox: Tuple[float, float, float, float] | None = None
        self.last_score: float = 0.0
        self.state: str = "Lost"
        self.frame_h: int | None = None
        self.frame_w: int | None = None

    def initialize(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> TrackerResult:
        """
        Initialize tracker from the first frame and initial target bbox.

        Args:
            frame: full frame image [H, W, 3]
            bbox: (x, y, w, h) in full-frame coordinates

        Returns:
            TrackerResult
        """
        self._validate_frame(frame)

        self.frame_h, self.frame_w = frame.shape[:2]

        bbox = self._clip_bbox_to_frame(bbox, self.frame_w, self.frame_h)
        template = self._crop_and_resize(frame, bbox, self.template_size)

        self.template_image = template
        self.current_bbox = bbox
        self.last_score = 1.0
        self.state = "Tracking"

        return TrackerResult(
            bbox=bbox,
            score=self.last_score,
            state=self.state,
            search_bbox=(
                int(round(bbox[0])),
                int(round(bbox[1])),
                int(round(bbox[2])),
                int(round(bbox[3])),
            ),
            grid_pos=(0, 0),
        )

    def track(self, frame: np.ndarray) -> TrackerResult:
        """
        Track target in the next frame.

        Args:
            frame: full frame image [H, W, 3]

        Returns:
            TrackerResult
        """
        self._validate_frame(frame)

        if self.template_image is None or self.current_bbox is None:
            raise RuntimeError("Tracker is not initialized. Call initialize(frame, bbox) first.")

        self.frame_h, self.frame_w = frame.shape[:2]

        search_bbox = self._build_search_region(
            self.current_bbox,
            self.frame_w,
            self.frame_h,
            scale=self.search_scale,
        )

        search_crop = self._crop_and_resize(frame, search_bbox, self.search_size)

        pred: PredictionResult = self.predictor.predict(
            template_image=self.template_image,
            search_image=search_crop,
        )

        new_bbox = self._decode_prediction_to_frame_bbox(
            pred=pred,
            search_bbox=search_bbox,
        )

        new_bbox = self._clip_bbox_to_frame(new_bbox, self.frame_w, self.frame_h)

        self.current_bbox = new_bbox
        self.last_score = pred.score
        self.state = self._score_to_state(pred.score)

        return TrackerResult(
            bbox=new_bbox,
            score=pred.score,
            state=self.state,
            search_bbox=(
                int(round(search_bbox[0])),
                int(round(search_bbox[1])),
                int(round(search_bbox[2])),
                int(round(search_bbox[3])),
            ),
            grid_pos=(pred.grid_x, pred.grid_y),
        )

    def _validate_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            raise ValueError("Frame is None.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected frame shape [H, W, 3], got {frame.shape}")

    def _score_to_state(self, score: float) -> str:
        if score >= self.tracking_threshold:
            return "Tracking"
        if score >= self.uncertain_threshold:
            return "Uncertain"
        return "Lost"

    def _build_search_region(
        self,
        bbox: Tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
        scale: float = 2.0,
    ) -> Tuple[float, float, float, float]:
        """
        Build a larger search region centered around the current bbox.
        """
        x, y, w, h = bbox

        cx = x + w / 2.0
        cy = y + h / 2.0

        search_w = max(20.0, w * scale)
        search_h = max(20.0, h * scale)

        search_x = cx - search_w / 2.0
        search_y = cy - search_h / 2.0

        return self._clip_bbox_to_frame(
            (search_x, search_y, search_w, search_h),
            frame_w,
            frame_h,
        )

    def _crop_and_resize(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        out_size: int,
    ) -> np.ndarray:
        """
        Crop bbox from frame and resize to square out_size x out_size.
        """
        x, y, w, h = bbox

        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        x2 = max(x1 + 1, min(x2, frame.shape[1]))
        y2 = max(y1 + 1, min(y2, frame.shape[0]))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            raise ValueError("Crop is empty after clipping.")

        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _clip_bbox_to_frame(
        self,
        bbox: Tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float, float]:
        """
        Clip x, y, w, h bbox to image bounds.
        """
        x, y, w, h = bbox

        x = float(x)
        y = float(y)
        w = max(1.0, float(w))
        h = max(1.0, float(h))

        x = max(0.0, min(x, frame_w - 1.0))
        y = max(0.0, min(y, frame_h - 1.0))

        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y

        w = max(1.0, w)
        h = max(1.0, h)

        return (x, y, w, h)

    def _decode_prediction_to_frame_bbox(
        self,
        pred: PredictionResult,
        search_bbox: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """
        Decode model prediction into full-frame bbox.

        Model predicts normalized [x, y, w, h] in the resized search crop.
        So:
        1) Convert normalized bbox to pixel coordinates inside 255x255 search crop
        2) Scale from resized crop space back to original search region in frame
        3) Add search region offset to get full-frame coordinates
        """
        sx, sy, sw, sh = search_bbox

        nx, ny, nw, nh = [float(v) for v in pred.bbox_raw]

        # normalized -> pixels in resized search crop
        px = nx * self.search_size
        py = ny * self.search_size
        pw = nw * self.search_size
        ph = nh * self.search_size

        # resized search crop -> original search region in frame
        scale_x = sw / self.search_size
        scale_y = sh / self.search_size

        frame_x = sx + px * scale_x
        frame_y = sy + py * scale_y
        frame_w = max(self.min_box_size, pw * scale_x)
        frame_h = max(self.min_box_size, ph * scale_y)

        return (frame_x, frame_y, frame_w, frame_h)