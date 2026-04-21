from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

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
    from app.traditional_tracker import TraditionalTracker

    backends: dict[str, type[BaseBackend]] = {
        "mock": MockBackend,
        "traditional_tracker": TraditionalTracker,
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

    def load(self, checkpoint: str | None) -> None:
        _ = checkpoint

    def reset(self) -> None:
        self.initialized = False

    def _state_from_confidence(self, confidence: float) -> str:
        if confidence < self.settings.lost_confidence_threshold:
            return "Lost"
        if confidence < self.settings.uncertain_confidence_threshold:
            return "Uncertain"
        return "Tracking"


class MockBackend(BaseBackend):
    def __init__(self, settings: TrackingSettings) -> None:
        super().__init__(settings)
        self.bbox: BBox | None = None
        self.step = 0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        _ = frame
        self.bbox = bbox
        self.step = 0
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
        self.bbox = (max(0, x + dx), max(0, y + dy), w, h)
        confidence = 0.92
        latency_ms = (perf_counter() - start) * 1000.0
        return TrackingResult(self.bbox, confidence, self._state_from_confidence(confidence), latency_ms)

    def reset(self) -> None:
        super().reset()
        self.bbox = None
        self.step = 0


