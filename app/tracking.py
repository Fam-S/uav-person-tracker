from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
import torch

from config import TrackingSettings
from data.crop_utils import (
    compute_crop_size,
    crop_and_resize,
    frame_to_tensor,
    xywh_to_center,
)


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
        self.search_size = 287
        self.output_size = 21
        self.anchor_stride = 8
        self.context_amount = 0.5
        self.template_scale = float(settings.template_crop_scale)
        # Original SiamAPN++ crop geometry fixes search scale from input sizes.
        # Do not use tracking.search_crop_scale here or train/inference targets drift.
        self.search_scale = self.search_size / self.template_size
        self.penalty_k = 0.079
        self.window_influence = 0.44
        self.lr = 0.31
        self.score_weights = (1.2, 1.0, 1.6)
        hanning = np.hanning(self.output_size)
        self.window = np.outer(hanning, hanning).flatten()
        self._last_bbox: BBox | None = None
        self.center_pos: np.ndarray | None = None
        self.size: np.ndarray | None = None

    def load(self, checkpoint: str | None) -> None:
        if not checkpoint:
            raise ValueError("SiamAPNBackend requires a checkpoint path")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from models import SiamAPNppMobileOne
        self.model = SiamAPNppMobileOne(feature_channels=192, pretrained_path=None, search_size=self.search_size).to(self.device)
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
            self.model.template(template_tensor)
        self._last_bbox = bbox
        self.center_pos = np.asarray([bbox[0] + (bbox[2] - 1) / 2.0, bbox[1] + (bbox[3] - 1) / 2.0], dtype=np.float32)
        self.size = np.asarray([bbox[2], bbox[3]], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.initialized = True

    @torch.no_grad()
    def track(self, frame: np.ndarray) -> TrackingResult:
        if not self.initialized or self.model is None or self._last_bbox is None or self.center_pos is None or self.size is None:
            return TrackingResult(None, 0.0, "Lost", 0.0)

        start = perf_counter()
        frame_h, frame_w = frame.shape[:2]
        search_center = (float(self.center_pos[0]), float(self.center_pos[1]))
        search_crop = crop_and_resize(
            frame,
            self._last_bbox,
            out_size=self.search_size,
            context_amount=self.context_amount,
            center_override=search_center,
            area_scale=self.search_scale,
        )
        search_tensor = frame_to_tensor(search_crop).unsqueeze(0).to(self.device)

        outputs = self.model.track(search_tensor)
        anchors = self._generate_anchor()
        score1 = self._convert_score(outputs["cls1"]) * self.score_weights[0]
        score2 = self._convert_score(outputs["cls2"]) * self.score_weights[1]
        score3 = outputs["cls3"].view(-1).detach().cpu().numpy() * self.score_weights[2]
        score = (score1 + score2 + score3) / 3.0
        pred_bbox = self._convert_bbox(outputs["loc"], anchors)

        s_z = compute_crop_size(self._last_bbox, context_amount=self.context_amount, area_scale=self.template_scale)
        scale_z = self.template_size / s_z
        s_c = self._change(self._sz(pred_bbox[2, :], pred_bbox[3, :]) / self._sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = self._change((self.size[0] / (self.size[1] + 1e-5)) / (pred_bbox[2, :] / (pred_bbox[3, :] + 1e-5)))
        penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
        pscore = penalty * score
        pscore = pscore * (1 - self.window_influence) + self.window * self.window_influence
        best_idx = int(np.argmax(pscore))
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.lr

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, (frame_h, frame_w))
        old_center = self.center_pos.copy()
        self.center_pos = np.asarray([cx, cy], dtype=np.float32)
        self.size = np.asarray([width, height], dtype=np.float32)
        new_bbox = (
            int(round(cx - width / 2.0)),
            int(round(cy - height / 2.0)),
            int(round(width)),
            int(round(height)),
        )

        latency_ms = (perf_counter() - start) * 1000.0

        measured = (self.center_pos - old_center).astype(np.float32)
        self.velocity = self.velocity * 0.7 + measured * 0.3
        self._last_bbox = new_bbox

        confidence = float(score[best_idx])
        return TrackingResult(new_bbox, confidence, self._state_from_confidence(confidence), latency_ms)

    @staticmethod
    def _change(r: np.ndarray) -> np.ndarray:
        return np.maximum(r, 1.0 / (r + 1e-5))

    @staticmethod
    def _sz(w: np.ndarray | float, h: np.ndarray | float) -> np.ndarray | float:
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    @staticmethod
    def _bbox_clip(cx: float, cy: float, width: float, height: float, boundary: tuple[int, int]) -> tuple[float, float, float, float]:
        cx = max(0.0, min(float(cx), float(boundary[1])))
        cy = max(0.0, min(float(cy), float(boundary[0])))
        width = max(10.0, min(float(width), float(boundary[1])))
        height = max(10.0, min(float(height), float(boundary[0])))
        return cx, cy, width, height

    def _generate_anchor(self) -> np.ndarray:
        if self.model is None or self.model.ranchors is None:
            raise RuntimeError("Model did not produce dynamic anchors.")
        mapp = self.model.ranchors
        size = self.output_size
        grid = np.linspace(0, size - 1, size)
        x = np.tile((self.anchor_stride * grid) - self.anchor_stride * (size - 1) / 2, size).reshape(-1)
        y = np.tile((self.anchor_stride * grid).reshape(-1, 1) - self.anchor_stride * (size - 1) / 2, size).reshape(-1)
        shap = (mapp[0] * (self.search_size // 4)).detach().cpu().numpy()
        xx = np.int16(np.tile(grid, size).reshape(-1))
        yy = np.int16(np.tile(grid.reshape(-1, 1), size).reshape(-1))
        w = shap[0, yy, xx] + shap[1, yy, xx]
        h = shap[2, yy, xx] + shap[3, yy, xx]
        x = x - shap[0, yy, xx] + w / 2
        y = y - shap[2, yy, xx] + h / 2
        anchor = np.zeros((size**2, 4), dtype=np.float32)
        anchor[:, 0] = x
        anchor[:, 1] = y
        anchor[:, 2] = w
        anchor[:, 3] = h
        return anchor

    @staticmethod
    def _convert_bbox(delta: torch.Tensor, anchor: np.ndarray) -> np.ndarray:
        delta_np = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).detach().cpu().numpy()
        delta_np[0, :] = delta_np[0, :] * anchor[:, 2] + anchor[:, 0]
        delta_np[1, :] = delta_np[1, :] * anchor[:, 3] + anchor[:, 1]
        delta_np[2, :] = np.exp(delta_np[2, :]) * anchor[:, 2]
        delta_np[3, :] = np.exp(delta_np[3, :]) * anchor[:, 3]
        return delta_np

    @staticmethod
    def _convert_score(score: torch.Tensor) -> np.ndarray:
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        return torch.softmax(score, dim=1).detach()[:, 1].cpu().numpy()

    def _smooth_bbox(self, bbox: BBox, frame_shape: tuple[int, int]) -> BBox:
        if self._last_bbox is None:
            return bbox

        prev_x, prev_y, prev_w, prev_h = [float(v) for v in self._last_bbox]
        pred_x, pred_y, pred_w, pred_h = [float(v) for v in bbox]

        prev_cx = prev_x + prev_w / 2.0
        prev_cy = prev_y + prev_h / 2.0
        pred_cx = pred_x + pred_w / 2.0
        pred_cy = pred_y + pred_h / 2.0

        max_step = max(4.0, max(prev_w, prev_h) * 0.5)
        dx = float(np.clip(pred_cx - prev_cx, -max_step, max_step))
        dy = float(np.clip(pred_cy - prev_cy, -max_step, max_step))
        smoothed_cx = prev_cx + dx * 0.5
        smoothed_cy = prev_cy + dy * 0.5

        smoothed_w = prev_w * 0.8 + pred_w * 0.2
        smoothed_h = prev_h * 0.8 + pred_h * 0.2

        frame_h, frame_w = frame_shape
        x = max(0.0, min(smoothed_cx - smoothed_w / 2.0, frame_w - 1.0))
        y = max(0.0, min(smoothed_cy - smoothed_h / 2.0, frame_h - 1.0))
        w = max(1.0, min(smoothed_w, frame_w - x))
        h = max(1.0, min(smoothed_h, frame_h - y))
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))

    def reset(self) -> None:
        super().reset()
        self._last_bbox = None
        self.center_pos = None
        self.size = None
