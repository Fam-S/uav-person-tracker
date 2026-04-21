from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from app.config import TrackingSettings


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class TrackingResult:
    bbox: BBox | None
    confidence: float
    state: str
    latency_ms: float


class TrackerBackend(Protocol):
    def load(self, checkpoint: str | None) -> None: ...
    def initialize(self, frame: np.ndarray, bbox: BBox) -> None: ...
    def track(self, frame: np.ndarray) -> TrackingResult: ...
    def reset(self) -> None: ...


def create_backend(settings: TrackingSettings) -> TrackerBackend:
    backends: dict[str, type[BaseBackend]] = {
        "mock": MockBackend,
        "csrt": CSRTBackend,
    }
    backend_class = backends.get(settings.backend)
    if backend_class is None:
        available = ", ".join(sorted(backends))
        raise ValueError(f"Unknown tracking backend '{settings.backend}'. Available: {available}")
    backend = backend_class(settings)
    backend.load(settings.checkpoint)
    return backend


class BaseBackend:
    def __init__(self, settings: TrackingSettings) -> None:
        self.settings = settings
        self.initialized = False
        self.velocity = np.zeros(2, dtype=np.float32)

    def load(self, checkpoint: str | None) -> None:
        _ = checkpoint

    def reset(self) -> None:
        self.initialized = False
        self.velocity = np.zeros(2, dtype=np.float32)

    def _state_from_confidence(self, confidence: float) -> str:
        if confidence < self.settings.lost_confidence_threshold:
            return "Lost"
        if confidence < self.settings.uncertain_confidence_threshold:
            return "Uncertain"
        return "Tracking"


class CSRTBackend(BaseBackend):
    def __init__(self, settings: TrackingSettings) -> None:
        super().__init__(settings)
        self._cv2 = cv2
        self._tracker = None
        self._last_bbox: BBox | None = None
        self._scale: float = 1.0

    def _create_tracker(self):
        if hasattr(self._cv2, "TrackerCSRT_create"):
            return self._cv2.TrackerCSRT_create()
        if hasattr(self._cv2, "TrackerCSRT") and hasattr(self._cv2.TrackerCSRT, "create"):
            return self._cv2.TrackerCSRT.create()
        legacy = getattr(self._cv2, "legacy", None)
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        raise RuntimeError("OpenCV build does not include the CSRT tracker")

    def _compute_scale(self, frame_w: int) -> float:
        max_w = self.settings.track_max_width
        if max_w <= 0 or frame_w <= max_w:
            return 1.0
        return max_w / frame_w

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = int(w * self._scale)
        new_h = int(h * self._scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _scale_bbox_down(self, bbox: BBox) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        s = self._scale
        return (int(x * s), int(y * s), max(1, int(w * s)), max(1, int(h * s)))

    def _scale_bbox_up(self, bbox: tuple) -> BBox:
        x, y, w, h = bbox
        s = self._scale
        return (int(x / s), int(y / s), max(1, int(w / s)), max(1, int(h / s)))

    def initialize(self, frame, bbox: BBox) -> None:
        self._scale = self._compute_scale(frame.shape[1])
        small = self._scale_frame(frame)
        small_bbox = self._scale_bbox_down(bbox)
        self._tracker = self._create_tracker()
        self._tracker.init(small, small_bbox)
        self._last_bbox = bbox
        self.velocity = np.zeros(2, dtype=np.float32)
        self.initialized = True

    def track(self, frame) -> TrackingResult:
        if not self.initialized or self._tracker is None:
            return TrackingResult(None, 0.0, "Lost", 0.0)
        start = perf_counter()
        small = self._scale_frame(frame)
        success, raw_bbox = self._tracker.update(small)
        latency_ms = (perf_counter() - start) * 1000.0
        if success:
            bbox = self._scale_bbox_up(raw_bbox)
            if self._last_bbox is not None:
                prev_cx = self._last_bbox[0] + self._last_bbox[2] / 2.0
                prev_cy = self._last_bbox[1] + self._last_bbox[3] / 2.0
                new_cx = bbox[0] + bbox[2] / 2.0
                new_cy = bbox[1] + bbox[3] / 2.0
                measured = np.array([new_cx - prev_cx, new_cy - prev_cy], dtype=np.float32)
                self.velocity = self.velocity * 0.7 + measured * 0.3
            else:
                self.velocity = np.zeros(2, dtype=np.float32)
            self._last_bbox = bbox
            return TrackingResult(bbox, 1.0, "Tracking", latency_ms)
        return TrackingResult(self._last_bbox, 0.0, "Lost", latency_ms)

    def reset(self) -> None:
        super().reset()
        self._tracker = None
        self._last_bbox = None
        self._scale = 1.0


class MockBackend(BaseBackend):
    def __init__(self, settings: TrackingSettings) -> None:
        super().__init__(settings)
        self.bbox: BBox | None = None
        self.step = 0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        _ = frame
        self.bbox = bbox
        self.step = 0
        self.velocity = np.zeros(2, dtype=np.float32)
        self.initialized = True

    def track(self, frame: np.ndarray) -> TrackingResult:
        _ = frame
        if not self.initialized or self.bbox is None:
            return TrackingResult(None, 0.0, "Lost", 0.0)

        start = perf_counter()
        x, y, w, h = self.bbox
        dx = (-2, -1, 0, 1, 2)[self.step % 5]
        dy = (0, 1, 2, 1, 0)[self.step % 5] - 1
        self.step += 1
        self.velocity = np.array([float(dx), float(dy)], dtype=np.float32)
        self.bbox = (max(0, x + dx), max(0, y + dy), w, h)
        confidence = 0.92
        latency_ms = (perf_counter() - start) * 1000.0
        return TrackingResult(self.bbox, confidence, self._state_from_confidence(confidence), latency_ms)

    def reset(self) -> None:
        super().reset()
        self.bbox = None
        self.step = 0
