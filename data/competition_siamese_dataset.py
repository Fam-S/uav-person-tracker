from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from torch import Tensor
from torch.utils.data import Dataset

from data.competition_data import SequenceRecord, load_sequences
from data.crop_utils import crop_and_resize, frame_to_tensor, project_box_to_crop, xywh_to_center


def _is_present(box_xywh: np.ndarray) -> bool:
    return bool(box_xywh[2] > 1.0 and box_xywh[3] > 1.0)


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
        self.epoch = 0
        self._bad_video_paths: set[Path] = set()
        self._readers: dict[Path, VideoReader] = {}

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

    def close(self) -> None:
        self._readers.clear()

    def __del__(self) -> None:
        self.close()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        sample_seed = self.seed + self.epoch * self.samples_per_epoch + int(index)
        return np.random.default_rng(sample_seed)

    def _get_reader(self, video_path: Path) -> VideoReader:
        reader = self._readers.get(video_path)
        if reader is None:
            try:
                reader = VideoReader(str(video_path), ctx=cpu(0))
            except Exception as exc:
                raise RuntimeError(f"Could not open video: {video_path}") from exc
            self._readers[video_path] = reader
        return reader

    def _release_reader(self, video_path: Path) -> None:
        self._readers.pop(video_path, None)

    def _load_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        reader = self._get_reader(video_path)
        try:
            frame_rgb = reader[int(frame_index)].asnumpy()
        except Exception as exc:
            self._release_reader(video_path)
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}.") from exc
        return np.ascontiguousarray(frame_rgb[:, :, ::-1])

    def _load_frame_pair(self, video_path: Path, template_index: int, search_index: int) -> tuple[np.ndarray, np.ndarray]:
        reader = self._get_reader(video_path)
        try:
            frames_rgb = reader.get_batch([int(template_index), int(search_index)]).asnumpy()
        except Exception as exc:
            self._release_reader(video_path)
            raise RuntimeError(
                f"Could not read frames {template_index} and {search_index} from {video_path}."
            ) from exc
        frames_bgr = np.ascontiguousarray(frames_rgb[:, :, :, ::-1])
        return frames_bgr[0], frames_bgr[1]

    def _sample_pair(self, rng: np.random.Generator) -> tuple[SequenceRecord, int, int]:
        available_sequences = [item for item in self.indexed_sequences if item.sequence.video_path not in self._bad_video_paths]
        if not available_sequences:
            raise RuntimeError("No readable training sequences remain after skipping bad videos.")

        indexed_sequence = available_sequences[int(rng.integers(len(available_sequences)))]
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
        max_attempts = max(8, len(self.indexed_sequences))
        last_error: Exception | None = None

        for _ in range(max_attempts):
            sequence, template_index, search_index = self._sample_pair(rng)

            template_box = sequence.gt_boxes_xywh[template_index]
            search_box = sequence.gt_boxes_xywh[search_index]

            try:
                template_frame, search_frame = self._load_frame_pair(
                    sequence.video_path,
                    template_index,
                    search_index,
                )
            except RuntimeError as exc:
                # Some competition videos are unreadable/corrupt on disk; skip them
                # for the rest of this worker instead of crashing the whole epoch.
                self._bad_video_paths.add(sequence.video_path)
                self._release_reader(sequence.video_path)
                last_error = exc
                continue

            try:
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
            except ValueError as exc:
                last_error = exc
                continue

            return {
                "template": frame_to_tensor(template_patch),
                "search": frame_to_tensor(search_patch),
                "search_bbox_xywh": torch.from_numpy(search_bbox_xywh.copy()),
                "seq_id": sequence.seq_id,
                "template_index": template_index,
                "search_index": search_index,
            }

        if last_error is not None:
            raise RuntimeError(f"Could not sample a valid training pair after {max_attempts} attempts.") from last_error
        raise RuntimeError("Could not sample a valid training pair.")
