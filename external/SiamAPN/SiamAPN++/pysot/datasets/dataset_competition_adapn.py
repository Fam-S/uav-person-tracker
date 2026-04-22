# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.core.config_adapn import cfg
from pysot.datasets.anchortarget_adapn import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.utils.bbox import Center, center2corner


logger = logging.getLogger("global")


class CompetitionSubDataset(object):
    def __init__(self, root, frame_range, num_use, start_idx):
        self.root = Path(root)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        self.labels = self._load_train_sequences()
        self.videos = list(self.labels.keys())
        self.num = len(self.videos)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.pick = self.shuffle()

    def _load_train_sequences(self):
        manifest_path = self.root / "metadata" / "contestant_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        train_entries = manifest["train"]
        labels = {}
        for seq_id, entry in train_entries.items():
            annotation_path = self.root / entry["annotation_path"]
            boxes = self._load_annotation_boxes(annotation_path)
            visible_frames = [idx for idx, box in enumerate(boxes) if box[2] > 0 and box[3] > 0]
            if len(visible_frames) < 2:
                continue

            frame_map = {}
            for frame_index in visible_frames:
                x, y, w, h = boxes[frame_index]
                frame_map["{:06d}".format(frame_index)] = [x, y, x + w, y + h]

            frame_map["frames"] = visible_frames
            labels[seq_id] = {
                "video_path": str(self.root / entry["video_path"]),
                "frames_data": frame_map,
            }
        return labels

    @staticmethod
    def _load_annotation_boxes(annotation_path):
        rows = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                rows.append([float(part) for part in parts])
        return np.asarray(rows, dtype=np.float32)

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, frame):
        frame_key = "{:06d}".format(frame)
        video_info = self.labels[video]
        return (video_info["video_path"], frame), video_info["frames_data"][frame_key]

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        frames = self.labels[video_name]["frames_data"]["frames"]
        template_pos = np.random.randint(0, len(frames))
        left = max(template_pos - self.frame_range, 0)
        right = min(template_pos + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_pos]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, template_frame), self.get_image_anno(video_name, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        frames = self.labels[video_name]["frames_data"]["frames"]
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, frame)


class CompetitionTrkDataset(Dataset):
    def __init__(self):
        super(CompetitionTrkDataset, self).__init__()
        self.anchor_target = AnchorTarget()
        self.sub_dataset = CompetitionSubDataset(
            root=cfg.DATASET.COMPETITION.ROOT,
            frame_range=cfg.DATASET.COMPETITION.FRAME_RANGE,
            num_use=cfg.DATASET.COMPETITION.NUM_USE,
            start_idx=0,
        )

        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR,
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR,
        )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.sub_dataset.num_use
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        while len(pick) < self.num:
            p = self.sub_dataset.pick[:]
            np.random.shuffle(p)
            pick += p
        return pick[:self.num]

    def __len__(self):
        return self.num

    @staticmethod
    def _read_frame(video_path, frame_index):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("Could not open video: {}".format(video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Could not read frame {} from {}".format(frame_index, video_path))
            return frame
        finally:
            capture.release()

    @staticmethod
    def _crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float32)
        return cv2.warpAffine(
            image,
            mapping,
            (out_sz, out_sz),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=padding,
        )

    @staticmethod
    def _pos_s_2_bbox(pos, size):
        return [pos[0] - size / 2, pos[1] - size / 2, pos[0] + size / 2, pos[1] + size / 2]

    def _crop_like_siamfc(self, image, bbox_xyxy, instanc_size=511):
        target_pos = [(bbox_xyxy[2] + bbox_xyxy[0]) / 2.0, (bbox_xyxy[3] + bbox_xyxy[1]) / 2.0]
        target_size = [bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]]
        wc_z = target_size[1] + 0.5 * sum(target_size)
        hc_z = target_size[0] + 0.5 * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z
        d_search = (instanc_size - cfg.TRAIN.EXEMPLAR_SIZE) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        avg_chans = np.mean(image, axis=(0, 1))
        return self._crop_hwc(image, self._pos_s_2_bbox(target_pos, s_x), instanc_size, avg_chans)

    @staticmethod
    def _get_bbox(image, shape):
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        imh, imw = image.shape[:2]
        cx, cy = imw // 2, imh // 2
        return center2corner(Center(cx, cy, w, h))

    def __getitem__(self, index):
        index = self.pick[index]
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        if neg:
            template_info = self.sub_dataset.get_random_target(index)
            search_info = self.sub_dataset.get_random_target()
        else:
            template_info, search_info = self.sub_dataset.get_positive_pair(index)

        (template_video_path, template_frame), template_bbox = template_info
        (search_video_path, search_frame), search_bbox = search_info

        template_frame_img = self._read_frame(template_video_path, template_frame)
        search_frame_img = self._read_frame(search_video_path, search_frame)

        template_image = self._crop_like_siamfc(template_frame_img, template_bbox)
        search_image = self._crop_like_siamfc(search_frame_img, search_bbox)

        template_box = self._get_bbox(template_image, template_bbox)
        search_box = self._get_bbox(search_image, search_bbox)

        template, _ = self.template_aug(template_image, template_box, cfg.TRAIN.EXEMPLAR_SIZE, gray=gray)
        search, bbox = self.search_aug(search_image, search_box, cfg.TRAIN.SEARCH_SIZE, gray=gray)

        labelcls2, labelxff, weightcls3, labelcls3, weightxff = self.anchor_target.get(
            bbox, cfg.TRAIN.OUTPUT_SIZE
        )

        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
            "template": template,
            "search": search,
            "bbox": np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.float32),
            "label_cls2": labelcls2,
            "labelxff": labelxff,
            "weightcls3": weightcls3,
            "labelcls3": labelcls3,
            "weightxff": weightxff,
        }
