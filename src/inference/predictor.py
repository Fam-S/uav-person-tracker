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
    """
    Run single-step inference on:
    - template image
    - search image

    Expected input images:
    - numpy arrays in HWC format
    - uint8 or float
    - BGR from OpenCV is acceptable as long as training used the same convention
    """

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

    def _resize_image(self, image: np.ndarray, out_size: int) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected image shape [H, W, 3], got {image.shape}"
            )

        resized = cv2.resize(image, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return resized

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image from HWC uint8/float -> BCHW float32 tensor in [0, 1].
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)  # [1, C, H, W]
        return tensor

    @torch.no_grad()
    def predict(
        self,
        template_image: np.ndarray,
        search_image: np.ndarray,
    ) -> PredictionResult:
        """
        Returns the best prediction from the model on one template/search pair.

        Output meaning:
        - score: max sigmoid score from cls logits
        - grid_x, grid_y: best cell on the classification map
        - bbox_raw: raw 4-value regression output at that best cell
        - cls_map: score map after sigmoid, shape [H, W]
        """
        template_image = self._resize_image(template_image, self.template_size)
        search_image = self._resize_image(search_image, self.search_size)

        template_tensor = self._to_tensor(template_image)
        search_tensor = self._to_tensor(search_image)

        outputs: dict[str, Any] = self.model(template_tensor, search_tensor)

        cls_logits = outputs["cls_logits"]   # [1, 1, H, W]
        bbox_pred = outputs["bbox_pred"]     # [1, 4, H, W]

        if cls_logits.ndim != 4 or bbox_pred.ndim != 4:
            raise ValueError(
                f"Unexpected output shapes: cls={tuple(cls_logits.shape)}, "
                f"bbox={tuple(bbox_pred.shape)}"
            )

        score_map = torch.sigmoid(cls_logits[0, 0])  # [H, W]

        flat_index = int(torch.argmax(score_map).item())
        map_h, map_w = score_map.shape

        grid_y = flat_index // map_w
        grid_x = flat_index % map_w

        best_score = float(score_map[grid_y, grid_x].item())
        bbox_raw = bbox_pred[0, :, grid_y, grid_x].detach().cpu().numpy().astype(np.float32)
        cls_map = score_map.detach().cpu().numpy().astype(np.float32)

        return PredictionResult(
            score=best_score,
            grid_x=grid_x,
            grid_y=grid_y,
            bbox_raw=bbox_raw,
            cls_map=cls_map,
        )