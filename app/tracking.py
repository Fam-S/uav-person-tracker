from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
import torch

from config import TrackingSettings
from data.crop_utils import crop_and_resize, frame_to_tensor, project_box_from_crop, xywh_to_center


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
        "siamapn": SiamAPNBackend,
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


class SiamAPNBackend(BaseBackend):
    def __init__(self, settings: TrackingSettings) -> None:
        super().__init__(settings)
        self.model = None
        self.device = None
        self.template_size = 127
        self.search_size = 255
        self.context_amount = 0.5
        self.template_scale = float(settings.template_crop_scale)
        self.search_scale = float(settings.search_crop_scale)
        self._last_bbox: BBox | None = None
        self._template_feat: dict | None = None

    def load(self, checkpoint: str | None) -> None:
        if not checkpoint:
            raise ValueError("SiamAPNBackend requires a checkpoint path")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from models import SiamAPNppMobileOne
        self.model = SiamAPNppMobileOne(feature_channels=192, pretrained_path=None).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        center = xywh_to_center(bbox)[:2]
        template_crop = crop_and_resize(
            frame,
            bbox,
            out_size=self.template_size,
            context_amount=self.context_amount,
            center_override=center,
            area_scale=self.template_scale,
        )
        template_tensor = frame_to_tensor(template_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self._template_feat = self.model._encode(template_tensor)
        self._last_bbox = bbox
        self.velocity = np.zeros(2, dtype=np.float32)
        self.initialized = True

    @torch.no_grad()
    def track(self, frame: np.ndarray) -> TrackingResult:
        if not self.initialized or self.model is None or self._template_feat is None:
            return TrackingResult(None, 0.0, "Lost", 0.0)

        start = perf_counter()
        frame_h, frame_w = frame.shape[:2]
        search_center = xywh_to_center(self._last_bbox)[:2]
        search_crop = crop_and_resize(
            frame,
            self._last_bbox,
            out_size=self.search_size,
            context_amount=self.context_amount,
            center_override=search_center,
            area_scale=self.search_scale,
        )
        search_tensor = frame_to_tensor(search_crop).unsqueeze(0).to(self.device)

        template_low, template_high = self._template_feat
        search_low, search_high = self.model.backbone(search_tensor)
        search_low = self.model.low_align(search_low)
        search_high = self.model.high_align(search_high)

        low_resp = self.model.correlation(template_low, search_low)
        high_resp = self.model.correlation(template_high, search_high)
        import torch.nn.functional as _F
        high_resp = _F.interpolate(high_resp, size=low_resp.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.model.fusion(torch.cat([low_resp, high_resp], dim=1))
        bbox_pred = self.model.reg_head(fused).flatten(1)
        bbox_pred = torch.cat([bbox_pred[:, :2], _F.softplus(bbox_pred[:, 2:])], dim=1)

        pred = bbox_pred[0].cpu().numpy()
        new_bbox = project_box_from_crop(
            pred,
            reference_box_xywh=self._last_bbox,
            out_size=self.search_size,
            context_amount=self.context_amount,
            center_override=search_center,
            area_scale=self.search_scale,
            frame_shape=(frame_h, frame_w),
        )

        latency_ms = (perf_counter() - start) * 1000.0

        prev_cx = self._last_bbox[0] + self._last_bbox[2] / 2.0
        prev_cy = self._last_bbox[1] + self._last_bbox[3] / 2.0
        new_cx = new_bbox[0] + new_bbox[2] / 2.0
        new_cy = new_bbox[1] + new_bbox[3] / 2.0
        measured = np.array([new_cx - prev_cx, new_cy - prev_cy], dtype=np.float32)
        self.velocity = self.velocity * 0.7 + measured * 0.3
        self._last_bbox = new_bbox

        return TrackingResult(new_bbox, 1.0, "Tracking", latency_ms)

    def reset(self) -> None:
        super().reset()
        self._template_feat = None
        self._last_bbox = None
