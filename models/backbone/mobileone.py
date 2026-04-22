from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone._mobileone import mobileone, reparameterize_model


class MobileOneS2Backbone(nn.Module):

    def __init__(
        self,
        pretrained_path: str | None = None,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_input = bool(normalize_input)

        base = mobileone(variant="s2", inference_mode=False)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            base.load_state_dict(state, strict=True)

        self.stage0 = base.stage0
        self.stage1 = base.stage1
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        del base

        self._low_channels = 256
        self._high_channels = 640

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    @property
    def low_channels(self) -> int:
        return self._low_channels

    @property
    def high_channels(self) -> int:
        return self._high_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_input:
            x = (x - self.pixel_mean) / self.pixel_std
        x = self.stage0(x)
        x = self.stage1(x)
        low = self.stage2(x)
        high = self.stage3(low)
        return low, high

    def reparameterize(self) -> None:
        for module in [self.stage0, self.stage1, self.stage2, self.stage3]:
            reparameterize_model(module)
