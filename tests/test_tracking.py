from __future__ import annotations

import numpy as np
import torch

from app.tracking import SiamAPNBackend
from config import TrackingSettings


class _FakeModel:
    def __init__(self) -> None:
        self.ranchors = None

    def template(self, x: torch.Tensor) -> None:
        _ = x

    def track(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _ = x
        self.ranchors = torch.full((1, 4, 21, 21), 0.2)
        cls = torch.zeros((1, 2, 21, 21), dtype=torch.float32)
        cls[:, 1, 10, 10] = 4.0
        cls3 = torch.zeros((1, 1, 21, 21), dtype=torch.float32)
        loc = torch.zeros((1, 4, 21, 21), dtype=torch.float32)
        return {"cls1": cls, "cls2": cls.clone(), "cls3": cls3, "loc": loc}


def test_siam_backend_reset_keeps_loaded_model_and_reinitializes():
    backend = SiamAPNBackend(
        TrackingSettings(
            backend="siamapn",
            checkpoint="dummy.pth",
            template_crop_scale=1.25,
            search_crop_scale=2.5,
        )
    )
    backend.model = _FakeModel()
    backend.device = torch.device("cpu")

    assert backend.template_scale == 1.25
    assert backend.search_scale == backend.search_size / backend.template_size

    backend.reset()

    frame = np.full((256, 256, 3), fill_value=128, dtype=np.uint8)
    backend.initialize(frame, (10, 12, 20, 24))

    assert backend.initialized is True
    assert backend.model is not None
    assert backend.device == torch.device("cpu")


def test_siam_backend_smooths_large_prediction_jumps(monkeypatch):
    backend = SiamAPNBackend(TrackingSettings(backend="siamapn", checkpoint="dummy.pth"))
    backend.model = _FakeModel()
    backend.device = torch.device("cpu")

    frame = np.full((256, 256, 3), fill_value=128, dtype=np.uint8)
    backend.initialize(frame, (10, 12, 20, 24))
    backend._last_bbox = (30, 34, 40, 44)
    backend.center_pos = np.asarray([50.0, 56.0], dtype=np.float32)
    backend.size = np.asarray([40.0, 44.0], dtype=np.float32)

    captured = []

    def fake_crop_and_resize(frame, box_xywh, out_size, context_amount, center_override=None, area_scale=1.0):
        _ = frame, context_amount, center_override, area_scale
        captured.append((tuple(box_xywh), out_size))
        return np.full((out_size, out_size, 3), fill_value=128, dtype=np.uint8)

    backend.model = _FakeModel()
    monkeypatch.setattr("app.tracking.crop_and_resize", fake_crop_and_resize)

    result = backend.track(frame)

    assert result.bbox is not None
    assert captured == [((30, 34, 40, 44), 287)]
    prev_center_x = 30 + 40 / 2.0
    new_center_x = result.bbox[0] + result.bbox[2] / 2.0
    assert abs(new_center_x - prev_center_x) <= 1.0
