from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.adapn_targets import AnchorTarget3
from models.adapn import APN, ClsAndLoc
from models.backbone import MobileOneS2Backbone
from models.losses import IOULoss, select_cross_entropy_loss, shaloss, weight_l1_loss


class SiamAPNppMobileOne(nn.Module):
    """Active SiamAPN++ + MobileOne-S2 implementation ported from the reference repo."""

    def __init__(
        self,
        feature_channels: int = 192,
        pretrained_path: str | None = None,
        normalize_input: bool = True,
        search_size: int = 287,
        output_size: int = 21,
        anchor_stride: int = 8,
    ) -> None:
        super().__init__()
        _ = feature_channels
        self.search_size = int(search_size)
        self.output_size = int(output_size)
        self.anchor_stride = int(anchor_stride)
        self.backbone = MobileOneS2Backbone(
            pretrained_path=pretrained_path,
            normalize_input=normalize_input,
        )
        self.grader = APN(apn_channels=384, head_channels=256)
        self.head = ClsAndLoc(channels=256, group_channels=32)
        self.anchor_target = AnchorTarget3(search_size=self.search_size, stride=self.anchor_stride)
        self.cls3loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IOULoss()
        self.ranchors: torch.Tensor | None = None
        self.zf: tuple[torch.Tensor, torch.Tensor] | None = None

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(x)

    def template(self, z: torch.Tensor) -> None:
        self.zf = self.backbone(z)

    def track(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.zf is None:
            raise RuntimeError("Call template() before track().")
        xf = self.backbone(x)
        xff, ress = self.grader(xf, self.zf)
        self.ranchors = xff
        cls1, cls2, cls3, loc = self.head(xf, self.zf, ress)
        return {"cls1": cls1, "cls2": cls2, "cls3": cls3, "loc": loc}

    @staticmethod
    def log_softmax(cls: torch.Tensor) -> torch.Tensor:
        batch, a2, height, width = cls.size()
        cls = cls.view(batch, 2, a2 // 2, height, width)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        return F.log_softmax(cls, dim=4)

    def getcenter(self, shape_pred: torch.Tensor) -> np.ndarray:
        batch = shape_pred.size(0)
        size = shape_pred.size(3)
        offset = self.search_size // 2 - self.anchor_stride * (size - 1) / 2
        grid = np.linspace(0, size - 1, size)
        x = np.tile((self.anchor_stride * grid + offset) - self.search_size // 2, size).reshape(-1)
        y = np.tile((self.anchor_stride * grid + offset).reshape(-1, 1) - self.search_size // 2, size).reshape(-1)
        shap = (shape_pred * (self.search_size // 4)).detach().cpu().numpy()
        xx = np.int16(np.tile(grid, size).reshape(-1))
        yy = np.int16(np.tile(grid.reshape(-1, 1), size).reshape(-1))
        w = shap[:, 0, yy, xx] + shap[:, 1, yy, xx]
        h = shap[:, 2, yy, xx] + shap[:, 3, yy, xx]
        x = x - shap[:, 0, yy, xx] + w / 2
        y = y - shap[:, 2, yy, xx] + h / 2
        anchor = np.zeros((batch, size**2, 4), dtype=np.float32)
        anchor[:, :, 0] = x + self.search_size // 2
        anchor[:, :, 1] = y + self.search_size // 2
        anchor[:, :, 2] = w
        anchor[:, :, 3] = h
        return anchor

    @staticmethod
    def _convert_bbox(delta: torch.Tensor, anchor: np.ndarray) -> torch.Tensor:
        delta = delta.contiguous().view(anchor.shape[0], 4, -1)
        anchor_tensor = torch.as_tensor(anchor, device=delta.device, dtype=delta.dtype)
        locc = torch.zeros_like(anchor_tensor)
        locc[:, :, 0] = delta[:, 0, :] * anchor_tensor[:, :, 2] + anchor_tensor[:, :, 0]
        locc[:, :, 1] = delta[:, 1, :] * anchor_tensor[:, :, 3] + anchor_tensor[:, :, 1]
        locc[:, :, 2] = torch.exp(delta[:, 2, :]) * anchor_tensor[:, :, 2]
        locc[:, :, 3] = torch.exp(delta[:, 3, :]) * anchor_tensor[:, :, 3]
        loc = torch.zeros_like(anchor_tensor)
        loc[:, :, 0] = locc[:, :, 0] - locc[:, :, 2] / 2
        loc[:, :, 1] = locc[:, :, 1] - locc[:, :, 3] / 2
        loc[:, :, 2] = locc[:, :, 0] + locc[:, :, 2] / 2
        loc[:, :, 3] = locc[:, :, 1] + locc[:, :, 3] / 2
        return loc

    def forward(self, template: torch.Tensor | dict[str, torch.Tensor], search: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if isinstance(template, dict):
            data = template
            template_tensor = data["template"]
            search_tensor = data["search"]
        else:
            if search is None:
                raise ValueError("search tensor is required when forward is called with tensors")
            data = None
            template_tensor = template
            search_tensor = search

        zf = self.backbone(template_tensor)
        xf = self.backbone(search_tensor)
        xff, ress = self.grader(xf, zf)
        cls1, cls2, cls3, loc = self.head(xf, zf, ress)
        outputs = {"cls1": cls1, "cls2": cls2, "cls3": cls3, "loc": loc, "ranchors": xff}
        if data is None or "bbox" not in data:
            return outputs

        bbox = data["bbox"]
        labelcls2 = data["label_cls2"]
        labelxff = data["labelxff"]
        labelcls3 = data["labelcls3"]
        weightxff = data["weightxff"]
        anchors = self.getcenter(xff)
        label_cls, label_loc, label_loc_weight = self.anchor_target.get(anchors, bbox, xff.size(3))

        cls1_log = self.log_softmax(cls1)
        cls2_log = self.log_softmax(cls2)
        cls_loss1 = select_cross_entropy_loss(cls1_log, label_cls)
        cls_loss2 = select_cross_entropy_loss(cls2_log, labelcls2)
        cls_loss3 = self.cls3loss(cls3, labelcls3)
        cls_loss = cls_loss1 + cls_loss2 + cls_loss3

        loc_loss1 = weight_l1_loss(loc, label_loc, label_loc_weight)
        pre_bbox = self._convert_bbox(loc, anchors)
        label_bbox = self._convert_bbox(label_loc, anchors)
        loc_loss2 = self.iou_loss(pre_bbox, label_bbox, label_loc_weight)
        loc_loss = loc_loss1 + loc_loss2
        shape_loss = shaloss(xff, labelxff, weightxff)
        total_loss = 3.0 * loc_loss + cls_loss + 2.0 * shape_loss
        outputs.update(
            {
                "total_loss": total_loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "shapeloss": shape_loss,
            }
        )
        return outputs
