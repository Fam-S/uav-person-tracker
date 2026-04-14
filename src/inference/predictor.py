from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class PredictionResult:
    score: float
    grid_x: int
    grid_y: int
    bbox_raw: np.ndarray
    cls_map: np.ndarray


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        template_size: int = 127,
        search_size: int = 255,
    ) -> None:
        self.model = model
        self.device = device
        self.template_size = int(template_size)
        self.search_size = int(search_size)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.float32: image = image.astype(np.float32)
        if image.max() > 1.0: image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, template_image: np.ndarray, search_image: np.ndarray) -> PredictionResult:
        # FIX: Convert BGR to RGB
        template_rgb = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
        search_rgb = cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)

        template_tensor = self._to_tensor(template_rgb)
        search_tensor = self._to_tensor(search_rgb)

        outputs: dict[str, Any] = self.model(template_tensor, search_tensor)

        cls_logits = outputs["cls_logits"]   # [1, 1, 5, 5]
        bbox_pred = outputs["bbox_pred"]     # [1, 4, 5, 5]

        score_map = torch.sigmoid(cls_logits[0, 0])  # [5, 5]
        
        flat_index = int(torch.argmax(score_map).item())
        map_h, map_w = score_map.shape
        grid_y = flat_index // map_w
        grid_x = flat_index % map_w
        best_score = float(score_map[grid_y, grid_x].item())
        
        # Extract raw offsets directly from 5x5
        bbox_raw = bbox_pred[0, :, grid_y, grid_x].detach().cpu().numpy().astype(np.float32)
        cls_map = score_map.detach().cpu().numpy().astype(np.float32)

        return PredictionResult(
            score=best_score,
            grid_x=grid_x,
            grid_y=grid_y,
            bbox_raw=bbox_raw,
            cls_map=cls_map,
        )