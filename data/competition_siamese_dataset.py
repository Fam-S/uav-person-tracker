from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.competition_data import SequenceRecord, load_sequences


def _is_present(box_xywh: np.ndarray) -> bool:
    return bool(box_xywh[2] > 1.0 and box_xywh[3] > 1.0)


def _xywh_to_center(box_xywh: np.ndarray) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return x + (w / 2.0), y + (h / 2.0), w, h


def _crop_and_resize(
    frame: np.ndarray,
    box_xywh: np.ndarray,
    out_size: int,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    center_x, center_y, box_w, box_h = _xywh_to_center(box_xywh)
    if center_override is not None:
        center_x, center_y = center_override

    context = context_amount * (box_w + box_h)
    crop_size = math.sqrt((box_w + context) * (box_h + context))
    crop_size = max(2.0, crop_size * float(area_scale))

    x1 = center_x - (crop_size / 2.0)
    y1 = center_y - (crop_size / 2.0)
    x2 = center_x + (crop_size / 2.0)
    y2 = center_y + (crop_size / 2.0)

    left_pad = max(0, int(math.floor(-x1)))
    top_pad = max(0, int(math.floor(-y1)))
    right_pad = max(0, int(math.ceil(x2 - frame_w)))
    bottom_pad = max(0, int(math.ceil(y2 - frame_h)))

    if left_pad or top_pad or right_pad or bottom_pad:
        frame = cv2.copyMakeBorder(
            frame,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            borderType=cv2.BORDER_REPLICATE,
        )

    x1_i, y1_i, x2_i, y2_i = _crop_bounds(
        frame_shape=(frame.shape[0], frame.shape[1]),
        box_xywh=box_xywh,
        context_amount=context_amount,
        center_override=center_override,
        area_scale=area_scale,
    )

    crop = frame[y1_i:y2_i, x1_i:x2_i]
    if crop.size == 0:
        raise ValueError("Crop produced an empty patch.")
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _project_box_to_crop(
    search_box_xywh: np.ndarray,
    reference_box_xywh: np.ndarray,
    out_size: int,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
) -> np.ndarray:
    center_x, center_y, ref_w, ref_h = _xywh_to_center(reference_box_xywh)
    if center_override is not None:
        center_x, center_y = center_override

    context = context_amount * (ref_w + ref_h)
    crop_size = math.sqrt((ref_w + context) * (ref_h + context))
    crop_size = max(2.0, crop_size * float(area_scale))
    scale = float(out_size) / crop_size

    x, y, w, h = [float(v) for v in search_box_xywh]
    crop_x1 = center_x - (crop_size / 2.0)
    crop_y1 = center_y - (crop_size / 2.0)

    projected = np.asarray(
        [
            (x - crop_x1) * scale,
            (y - crop_y1) * scale,
            w * scale,
            h * scale,
        ],
        dtype=np.float32,
    )
    return np.clip(projected, a_min=0.0, a_max=float(out_size))


def _frame_to_tensor(frame_bgr: np.ndarray) -> Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().float()
    return tensor / 255.0


def _load_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}.")
        return frame
    finally:
        capture.release()


@dataclass(frozen=True)
class SequenceIndex:
    sequence: SequenceRecord
    valid_indices: np.ndarray


@dataclass(frozen=True)
class SamplePair:
    sequence: SequenceRecord
    template_index: int
    search_index: int


def _crop_bounds(
    frame_shape: tuple[int, int],
    box_xywh: np.ndarray,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape
    center_x, center_y, box_w, box_h = _xywh_to_center(box_xywh)
    if center_override is not None:
        center_x, center_y = center_override

    context = context_amount * (box_w + box_h)
    crop_size = math.sqrt((box_w + context) * (box_h + context))
    crop_size = max(2.0, crop_size * float(area_scale))

    x1 = center_x - (crop_size / 2.0)
    y1 = center_y - (crop_size / 2.0)
    x2 = center_x + (crop_size / 2.0)
    y2 = center_y + (crop_size / 2.0)

    left_pad = max(0, int(math.floor(-x1)))
    top_pad = max(0, int(math.floor(-y1)))
    right_pad = max(0, int(math.ceil(x2 - frame_w)))
    bottom_pad = max(0, int(math.ceil(y2 - frame_h)))

    x1 += left_pad
    x2 += left_pad
    y1 += top_pad
    y2 += top_pad

    padded_w = frame_w + left_pad + right_pad
    padded_h = frame_h + top_pad + bottom_pad

    x1_i = max(0, min(int(round(x1)), padded_w))
    y1_i = max(0, min(int(round(y1)), padded_h))
    x2_i = max(0, min(int(round(x2)), padded_w))
    y2_i = max(0, min(int(round(y2)), padded_h))
    return x1_i, y1_i, x2_i, y2_i


def _can_crop(
    frame_shape: tuple[int, int],
    box_xywh: np.ndarray,
    context_amount: float,
    center_override: tuple[float, float] | None = None,
    area_scale: float = 1.0,
) -> bool:
    x1_i, y1_i, x2_i, y2_i = _crop_bounds(
        frame_shape=frame_shape,
        box_xywh=box_xywh,
        context_amount=context_amount,
        center_override=center_override,
        area_scale=area_scale,
    )
    return x2_i > x1_i and y2_i > y1_i


class CompetitionSiameseDataset(Dataset[dict[str, Tensor | str | int]]):
    """Minimal pair-sampling dataset for the competition train split."""

    def __init__(
        self,
        raw_root: str | Path,
        sequences: list[SequenceRecord] | None = None,
        template_size: int = 127,
        search_size: int = 255,
        context_amount: float = 0.5,
        samples_per_epoch: int = 512,
        frame_range: int = 100,
        translation_jitter: float = 0.0,
        scale_jitter: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.raw_root = Path(raw_root)
        self.template_size = int(template_size)
        self.search_size = int(search_size)
        self.context_amount = float(context_amount)
        self.samples_per_epoch = int(samples_per_epoch)
        self.frame_range = int(frame_range)
        self.translation_jitter = float(translation_jitter)
        self.scale_jitter = float(scale_jitter)
        self.seed = int(seed)

        if sequences is None:
            sequences = load_sequences(self.raw_root, "train")
        indexed_sequences = self._build_index(sequences)
        if not indexed_sequences:
            raise ValueError("No train sequences contain at least two visible target frames.")
        self.sample_pairs = self._build_sample_pairs(indexed_sequences)
        if not self.sample_pairs:
            raise ValueError("No valid train frame pairs remain after crop validation.")

    @staticmethod
    def _build_index(sequences: list[SequenceRecord]) -> list[SequenceIndex]:
        indexed: list[SequenceIndex] = []
        for sequence in sequences:
            if sequence.gt_boxes_xywh is None:
                continue
            valid_indices = np.flatnonzero(
                np.asarray([_is_present(box) for box in sequence.gt_boxes_xywh], dtype=np.bool_)
            )
            if valid_indices.size < 2:
                continue
            indexed.append(SequenceIndex(sequence=sequence, valid_indices=valid_indices))
        return indexed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng_for_index(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(index))

    def _build_sample_pairs(self, indexed_sequences: list[SequenceIndex]) -> list[SamplePair]:
        sample_pairs: list[SamplePair] = []
        scale_factors = [1.0]
        if self.scale_jitter > 0:
            scale_factors = [
                max(0.5, 1.0 - self.scale_jitter),
                1.0 + self.scale_jitter,
            ]

        for indexed_sequence in indexed_sequences:
            frame_shape = _load_frame(indexed_sequence.sequence.video_path, 0).shape[:2]
            valid_indices = indexed_sequence.valid_indices

            for template_pos in range(valid_indices.size - 1):
                template_index = int(valid_indices[template_pos])
                template_box = indexed_sequence.sequence.gt_boxes_xywh[template_index]
                if not _can_crop(frame_shape, template_box, self.context_amount, area_scale=1.0):
                    continue

                search_candidates = valid_indices[template_pos + 1 :]
                if search_candidates.size == 0:
                    search_candidates = valid_indices[template_pos : template_pos + 1]

                max_search = template_index + self.frame_range
                within_range = search_candidates[search_candidates <= max_search]
                if within_range.size > 0:
                    search_candidates = within_range

                for search_index in search_candidates:
                    search_box = indexed_sequence.sequence.gt_boxes_xywh[int(search_index)]
                    search_center = _xywh_to_center(search_box)[:2]

                    centers = [search_center]
                    if self.translation_jitter > 0:
                        dx = self.translation_jitter * float(search_box[2])
                        dy = self.translation_jitter * float(search_box[3])
                        centers = [
                            (search_center[0] - dx, search_center[1] - dy),
                            (search_center[0] - dx, search_center[1] + dy),
                            (search_center[0] + dx, search_center[1] - dy),
                            (search_center[0] + dx, search_center[1] + dy),
                        ]

                    is_valid = True
                    for center in centers:
                        for scale_factor in scale_factors:
                            if not _can_crop(
                                frame_shape,
                                template_box,
                                self.context_amount,
                                center_override=center,
                                area_scale=2.0 * scale_factor,
                            ):
                                is_valid = False
                                break
                        if not is_valid:
                            break

                    if is_valid:
                        sample_pairs.append(
                            SamplePair(
                                sequence=indexed_sequence.sequence,
                                template_index=template_index,
                                search_index=int(search_index),
                            )
                        )

        return sample_pairs

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        rng = self._rng_for_index(index)
        pair = self.sample_pairs[int(rng.integers(len(self.sample_pairs)))]
        sequence = pair.sequence
        template_index = pair.template_index
        search_index = pair.search_index

        template_box = sequence.gt_boxes_xywh[template_index]
        search_box = sequence.gt_boxes_xywh[search_index]

        template_frame = _load_frame(sequence.video_path, template_index)
        search_frame = _load_frame(sequence.video_path, search_index)

        search_center = _xywh_to_center(search_box)[:2]

        scale_factor = 1.0
        if self.scale_jitter > 0:
            scale_factor = 1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter)
            scale_factor = max(0.5, scale_factor)

        template_patch = _crop_and_resize(
            template_frame,
            template_box,
            out_size=self.template_size,
            context_amount=self.context_amount,
            area_scale=1.0,
        )

        tx_jitter, ty_jitter = 0.0, 0.0
        if self.translation_jitter > 0:
            tx_jitter = rng.uniform(-self.translation_jitter, self.translation_jitter)
            ty_jitter = rng.uniform(-self.translation_jitter, self.translation_jitter)
        jittered_center = (
            search_center[0] + tx_jitter * search_box[2],
            search_center[1] + ty_jitter * search_box[3],
        )

        search_patch = _crop_and_resize(
            search_frame,
            template_box,
            out_size=self.search_size,
            context_amount=self.context_amount,
            center_override=jittered_center,
            area_scale=2.0 * scale_factor,
        )
        search_bbox_xywh = _project_box_to_crop(
            search_box_xywh=search_box,
            reference_box_xywh=template_box,
            out_size=self.search_size,
            context_amount=self.context_amount,
            center_override=jittered_center,
            area_scale=2.0 * scale_factor,
        )

        return {
            "template": _frame_to_tensor(template_patch),
            "search": _frame_to_tensor(search_patch),
            "search_bbox_xywh": torch.from_numpy(search_bbox_xywh.copy()),
            "seq_id": sequence.seq_id,
            "template_index": template_index,
            "search_index": search_index,
        }
