from __future__ import annotations

import numpy as np
import torch


def _select(position: tuple[np.ndarray, ...], keep_num: int) -> tuple[tuple[np.ndarray, ...], int]:
    num = position[0].shape[0]
    if num <= keep_num:
        return position, num
    selected = np.arange(num)
    np.random.shuffle(selected)
    selected = selected[:keep_num]
    return tuple(p[selected] for p in position), keep_num


def _iou(boxes: list[np.ndarray], target: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes
    tx1, ty1, tx2, ty2 = [float(v) for v in target]
    inter_w = np.maximum(0.0, np.minimum(x2, tx2) - np.maximum(x1, tx1))
    inter_h = np.maximum(0.0, np.minimum(y2, ty2) - np.maximum(y1, ty1))
    inter = inter_w * inter_h
    area_boxes = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_target = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    return inter / np.maximum(area_boxes + area_target - inter, 1e-6)


class AnchorTarget:
    def __init__(self, search_size: int = 287, stride: int = 8) -> None:
        self.search_size = int(search_size)
        self.stride = int(stride)
        self.weightxffrange = 8

    def get(self, bbox_xyxy: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        offset = self.search_size // 2 - self.stride * (size - 1) / 2
        labelcls2 = np.zeros((1, size, size), dtype=np.float32) - 1
        pre = (self.stride * np.linspace(0, size - 1, size) + offset).reshape(-1, 1)
        points = np.zeros((size**2, 2), dtype=np.float32)
        points[:, 0] = np.maximum(0, np.tile(pre, size).T.reshape(-1))
        points[:, 1] = np.maximum(0, np.tile(pre, size).reshape(-1))

        labelxff = np.zeros((4, size, size), dtype=np.float32)
        weightcls3 = np.zeros((1, size, size), dtype=np.float32)
        labelcls3 = np.zeros((1, size, size), dtype=np.float32)
        weightxff = np.zeros((1, size, size), dtype=np.float32)
        target = np.asarray(bbox_xyxy, dtype=np.float32)

        index2 = np.int32((target - offset) / self.stride)
        w = int(index2[2] - index2[0])
        h = int(index2[3] - index2[1])
        weightxff[
            0,
            max(0, index2[1] - h // self.weightxffrange) : min(size, index2[3] + 1 + h // self.weightxffrange),
            max(0, index2[0] - w // self.weightxffrange) : min(size, index2[2] + 1 + w // self.weightxffrange),
        ] = 1

        index = np.minimum(size - 1, np.maximum(0, np.int32((target - offset) / self.stride)))
        w = int(index[2] - index[0]) + 1
        h = int(index[3] - index[1]) + 1
        weightcls3[0, index[1] : index[3] + 1, index[0] : index[2] + 1] = 0
        weightcls3[0, index[1] + h // 4 : index[3] + 1 - h // 4, index[0] + w // 4 : index[2] + 1 - w // 4] = 1

        for yy in np.arange(index[1], index[3] + 1):
            for xx in np.arange(index[0], index[2] + 1):
                l1 = np.minimum(yy - index[1], index[3] - yy) / (np.maximum(yy - index[1], index[3] - yy) + 1e-4)
                l2 = np.minimum(xx - index[0], index[2] - xx) / (np.maximum(xx - index[0], index[2] - xx) + 1e-4)
                labelcls3[0, yy, xx] = weightcls3[0, yy, xx] * np.sqrt(l1 * l2)

        labelxff[0, :, :] = (points[:, 0] - target[0]).reshape(size, size)
        labelxff[1, :, :] = (target[2] - points[:, 0]).reshape(size, size)
        labelxff[2, :, :] = (points[:, 1] - target[1]).reshape(size, size)
        labelxff[3, :, :] = (target[3] - points[:, 1]).reshape(size, size)
        labelxff = labelxff / (self.search_size // 4)

        ww = int(index[2] - index[0]) + 1
        hh = int(index[3] - index[1]) + 1
        labelcls2[0, index[1] : index[3] + 1, index[0] : index[2] + 1] = -2
        labelcls2[0, index[1] + hh // 4 : index[3] + 1 - hh // 4, index[0] + ww // 4 : index[2] + 1 - ww // 4] = 1
        neg2, _ = _select(np.where(labelcls2.squeeze() == -1), int((labelcls2 == 1).sum() * 3))
        if len(neg2[0]) > 0:
            labelcls2[:, neg2[0], neg2[1]] = 0
        return labelcls2, labelxff.astype(np.float32), weightcls3, labelcls3, weightxff


class AnchorTarget3:
    def __init__(self, search_size: int = 287, stride: int = 8) -> None:
        self.search_size = int(search_size)
        self.stride = int(stride)
        self.pos_num = 16
        self.total_num = 64
        self.neg_num = 16
        self.labelcls2range1 = 8
        self.labelcls2range2 = 6
        self.labelcls2range3 = 4

    def get(
        self,
        anchors: np.ndarray,
        targets: torch.Tensor,
        size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(targets.shape[0])
        anchor_num = 1
        cls = -1 * np.ones((batch, anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((batch, 4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((batch, anchor_num, size, size), dtype=np.float32)
        offset = self.search_size // 2 - self.stride * (size - 1) / 2

        targets_np = targets.detach().cpu().numpy()
        for i in range(batch):
            anchor = anchors[i]
            target = targets_np[i]
            tcx = (target[0] + target[2]) / 2
            tcy = (target[1] + target[3]) / 2
            tw = target[2] - target[0]
            th = target[3] - target[1]

            cx = anchor[:, 0].reshape(1, size, size)
            cy = anchor[:, 1].reshape(1, size, size)
            w = anchor[:, 2].reshape(1, size, size)
            h = anchor[:, 3].reshape(1, size, size)
            x1 = cx - w * 0.5
            y1 = cy - h * 0.5
            x2 = cx + w * 0.5
            y2 = cy + h * 0.5

            index = np.minimum(size - 1, np.maximum(0, np.int32((target - offset) / self.stride)))
            ww = int(index[2] - index[0]) + 1
            hh = int(index[3] - index[1]) + 1
            labelcls2 = np.zeros((1, size, size), dtype=np.float32) - 2
            labelcls2[
                0,
                max(0, index[1] - hh // self.labelcls2range1) : min(size, index[3] + 1 + hh // self.labelcls2range1),
                max(0, index[0] - ww // self.labelcls2range1) : min(size, index[2] + 1 + ww // self.labelcls2range1),
            ] = -1
            labelcls2[0, index[1] : index[3] + 1, index[0] : index[2] + 1] = 0
            labelcls2[
                0,
                index[1] + hh // self.labelcls2range2 : index[3] - hh // self.labelcls2range2 + 1,
                index[0] + ww // self.labelcls2range2 : index[2] - ww // self.labelcls2range2 + 1,
            ] = 0.5
            labelcls2[
                0,
                index[1] + hh // self.labelcls2range3 : index[3] - hh // self.labelcls2range3 + 1,
                index[0] + ww // self.labelcls2range3 : index[2] - ww // self.labelcls2range3 + 1,
            ] = 1

            overlap = _iou([x1, y1, x2, y2], target)
            cls_pos = np.where(overlap > 0.86)
            cls_neg = np.where(overlap <= 0.6)
            cls_pos, _ = _select(cls_pos, self.pos_num)
            cls_neg, _ = _select(cls_neg, self.total_num - self.pos_num)
            cls[i][cls_pos] = 1
            cls[i][cls_neg] = 0

            reg_pos = np.where((overlap > 0.83) | ((overlap > 0.8) & (labelcls2 >= 0.5)))
            reg_neg = np.where(overlap <= 0.6)
            reg_pos, pos_num = _select(reg_pos, self.pos_num)
            reg_neg, _ = _select(reg_neg, self.total_num - self.pos_num)
            if anchor[:, 2].min() > 0 and anchor[:, 3].min() > 0:
                delta[i][0] = (tcx - cx) / (w + 1e-6)
                delta[i][1] = (tcy - cy) / (h + 1e-6)
                delta[i][2] = np.log(tw / (w + 1e-6) + 1e-6)
                delta[i][3] = np.log(th / (h + 1e-6) + 1e-6)
                delta_weight[i][reg_pos] = 1.0 / (pos_num + 1e-6)
                delta_weight[i][reg_neg] = 0

        device = targets.device
        return (
            torch.from_numpy(cls).to(device=device),
            torch.from_numpy(delta).to(device=device),
            torch.from_numpy(delta_weight).to(device=device),
        )
