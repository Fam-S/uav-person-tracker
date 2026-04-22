from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
import torch

from config import TrackingSettings


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
        self.search_scale = 2.0
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

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0

    def _crop_and_resize(self, frame: np.ndarray, box_xywh: tuple, out_size: int, center: tuple, area_scale: float) -> np.ndarray:
        x, y, w, h = [float(v) for v in box_xywh]
        cx, cy = center
        context = self.context_amount * (w + h)
        crop_size = max(2.0, math.sqrt((w + context) * (h + context)) * area_scale)

        x1 = cx - crop_size / 2.0
        y1 = cy - crop_size / 2.0
        x2 = cx + crop_size / 2.0
        y2 = cy + crop_size / 2.0

        fh, fw = frame.shape[:2]
        lp = max(0, int(math.floor(-x1)))
        tp = max(0, int(math.floor(-y1)))
        rp = max(0, int(math.ceil(x2 - fw)))
        bp = max(0, int(math.ceil(y2 - fh)))
        if lp or tp or rp or bp:
            frame = cv2.copyMakeBorder(frame, tp, bp, lp, rp, borderType=cv2.BORDER_REPLICATE)
            x1 += lp; x2 += lp; y1 += tp; y2 += tp

        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            crop = frame
        return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    def _box_center(self, box_xywh: tuple) -> tuple[float, float]:
        x, y, w, h = [float(v) for v in box_xywh]
        return x + w / 2.0, y + h / 2.0

    def _crop_to_bbox(self, pred_xywh: np.ndarray, center: tuple, crop_size: float, frame_h: int, frame_w: int) -> BBox:
        x, y, w, h = pred_xywh
        scale = crop_size / self.search_size
        cx, cy = center
        fx = cx - crop_size / 2.0
        fy = cy - crop_size / 2.0
        rx = fx + x * scale
        ry = fy + y * scale
        rw = w * scale
        rh = h * scale
        rx = max(0, min(rx, frame_w - 1))
        ry = max(0, min(ry, frame_h - 1))
        rw = max(1, min(rw, frame_w - rx))
        rh = max(1, min(rh, frame_h - ry))
        return (int(rx), int(ry), int(rw), int(rh))

    def _compute_search_crop_params(self, box_xywh: tuple):
        x, y, w, h = [float(v) for v in box_xywh]
        cx, cy = x + w / 2.0, y + h / 2.0
        context = self.context_amount * (w + h)
        crop_size = max(2.0, math.sqrt((w + context) * (h + context)) * self.search_scale)
        return cx, cy, crop_size

    @torch.no_grad()
    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        x, y, w, h = [float(v) for v in bbox]
        center = (x + w / 2.0, y + h / 2.0)
        template_crop = self._crop_and_resize(frame, bbox, self.template_size, center, 1.0)
        template_tensor = self._frame_to_tensor(template_crop).unsqueeze(0).to(self.device)

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
        fh, fw = frame.shape[:2]
        cx, cy, crop_size = self._compute_search_crop_params(self._last_bbox)
        search_crop = self._crop_and_resize(frame, self._last_bbox, self.search_size, (cx, cy), self.search_scale)
        search_tensor = self._frame_to_tensor(search_crop).unsqueeze(0).to(self.device)

        template_low, template_high = self._template_feat
        from models.siamapn import DepthwiseCrossCorrelation
        correlation = DepthwiseCrossCorrelation()

        from models.siamapn import FeatureAlign
        search_low, search_high = self.model.backbone(search_tensor)
        search_low = self.model.low_align(search_low)
        search_high = self.model.high_align(search_high)

        low_resp = correlation(template_low, search_low)
        high_resp = correlation(template_high, search_high)
        import torch.nn.functional as _F
        high_resp = _F.interpolate(high_resp, size=low_resp.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.model.fusion(torch.cat([low_resp, high_resp], dim=1))
        bbox_pred = self.model.reg_head(fused).flatten(1)
        bbox_pred = torch.cat([bbox_pred[:, :2], _F.softplus(bbox_pred[:, 2:])], dim=1)

        pred = bbox_pred[0].cpu().numpy()
        new_bbox = self._crop_to_bbox(pred, (cx, cy), crop_size, fh, fw)

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
        self.model = None
        self.device = None
        self._template_feat = None
        self._last_bbox = None
