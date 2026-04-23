from __future__ import annotations

import numpy as np
import torch

from app.tracking import SiamAPNBackend
from config import TrackingSettings


class _FakeModel:
    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x


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
    assert backend.search_scale == 2.5

    backend.reset()

    frame = np.full((64, 64, 3), fill_value=128, dtype=np.uint8)
    backend.initialize(frame, (10, 12, 20, 24))

    assert backend.initialized is True
    assert backend.model is not None
    assert backend.device == torch.device("cpu")
