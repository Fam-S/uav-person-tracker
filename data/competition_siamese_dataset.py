from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.competition_data import SequenceRecord, load_sequences
from data.crop_utils import crop_and_resize, frame_to_tensor, project_box_to_crop, xywh_to_center


def _is_present(box_xywh: np.ndarray) -> bool:
    return bool(box_xywh[2] > 1.0 and box_xywh[3] > 1.0)


def _load_frame(video_path: Path, frame_index: int) -> np.ndarray:
    import cv2

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


class CompetitionSiameseDataset(Dataset[dict[str, Tensor | str | int]]):
    """Minimal pair-sampling dataset for the competition train split."""

    def __init__(
        self,
        raw_root: str | Path,
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

        sequences = load_sequences(self.raw_root, "train")
        self.indexed_sequences = self._build_index(sequences)
        if not self.indexed_sequences:
            raise ValueError("No train sequences contain at least two visible target frames.")

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

    def _sample_pair(self, rng: np.random.Generator) -> tuple[SequenceRecord, int, int]:
        indexed_sequence = self.indexed_sequences[int(rng.integers(len(self.indexed_sequences)))]
        valid_indices = indexed_sequence.valid_indices

        template_pos = int(rng.integers(0, valid_indices.size - 1))
        template_index = int(valid_indices[template_pos])

        search_candidates = valid_indices[template_pos + 1 :]
        if search_candidates.size == 0:
            search_candidates = valid_indices[template_pos : template_pos + 1]

        max_search = template_index + self.frame_range
        within_range = search_candidates[search_candidates <= max_search]
        if within_range.size > 0:
            search_candidates = within_range

        search_index = int(search_candidates[int(rng.integers(search_candidates.size))])
        return indexed_sequence.sequence, template_index, search_index

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        rng = self._rng_for_index(index)
        sequence, template_index, search_index = self._sample_pair(rng)

        template_box = sequence.gt_boxes_xywh[template_index]
        search_box = sequence.gt_boxes_xywh[search_index]

        template_frame = _load_frame(sequence.video_path, template_index)
        search_frame = _load_frame(sequence.video_path, search_index)

        search_center = xywh_to_center(search_box)[:2]

        scale_factor = 1.0
        if self.scale_jitter > 0:
            scale_factor = 1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter)
            scale_factor = max(0.5, scale_factor)

        template_patch = crop_and_resize(
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

        search_patch = crop_and_resize(
            search_frame,
            template_box,
            out_size=self.search_size,
            context_amount=self.context_amount,
            center_override=jittered_center,
            area_scale=2.0 * scale_factor,
        )
        search_bbox_xywh = project_box_to_crop(
            search_box_xywh=search_box,
            reference_box_xywh=template_box,
            out_size=self.search_size,
            context_amount=self.context_amount,
            center_override=jittered_center,
            area_scale=2.0 * scale_factor,
        )

        return {
            "template": frame_to_tensor(template_patch),
            "search": frame_to_tensor(search_patch),
            "search_bbox_xywh": torch.from_numpy(search_bbox_xywh.copy()),
            "seq_id": sequence.seq_id,
            "template_index": template_index,
            "search_index": search_index,
        }
