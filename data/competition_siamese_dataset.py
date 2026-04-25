from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import cv2
from torch import Tensor
from torch.utils.data import Dataset

from decord import logging as decord_logging

from data.competition_data import SequenceRecord, load_sequences
from data.adapn_targets import AnchorTarget
from data.crop_utils import (
    crop_and_resize,
    frame_to_tensor,
    project_box_to_crop,
    xywh_to_center,
)

if TYPE_CHECKING:
    from decord import VideoReader


decord_logging.set_level(decord_logging.QUIET)


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
        output_size: int = 21,
        context_amount: float = 0.5,
        samples_per_epoch: int = 512,
        frame_range: int = 100,
        translation_jitter: float = 0.0,
        scale_jitter: float = 0.0,
        color_jitter_prob: float = 0.0,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        saturation_jitter: float = 0.0,
        grayscale_prob: float = 0.0,
        blur_prob: float = 0.0,
        noise_prob: float = 0.0,
        horizontal_flip_prob: float = 0.0,
        template_trim_jitter: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.raw_root = Path(raw_root)
        self.template_size = int(template_size)
        self.search_size = int(search_size)
        self.output_size = int(output_size)
        self.context_amount = float(context_amount)
        self.search_area_scale = float(self.search_size) / float(self.template_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.frame_range = int(frame_range)
        self.translation_jitter = float(translation_jitter)
        self.scale_jitter = float(scale_jitter)
        self.color_jitter_prob = float(color_jitter_prob)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.saturation_jitter = float(saturation_jitter)
        self.grayscale_prob = float(grayscale_prob)
        self.blur_prob = float(blur_prob)
        self.noise_prob = float(noise_prob)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.template_trim_jitter = float(template_trim_jitter)
        self.seed = int(seed)
        self.epoch = 0
        self.anchor_target = AnchorTarget(search_size=self.search_size, stride=8)
        self._bad_video_paths: set[Path] = self._load_known_bad_video_paths()
        # Keep only a tiny number of live decoders per worker. Random video sampling
        # plus DataLoader prefetch can otherwise accumulate many open readers and
        # trigger the OOM killer on constrained environments like Kaggle.
        self._max_open_readers = 2
        self._readers: OrderedDict[Path, VideoReader] = OrderedDict()

        sequences = load_sequences(self.raw_root, "train")
        self.indexed_sequences = self._build_index(sequences)
        if not self.indexed_sequences:
            raise ValueError("No train sequences contain at least two visible target frames.")

    def _load_known_bad_video_paths(self) -> set[Path]:
        bad_list_path = self.raw_root / "metadata" / "bad_videos.txt"
        if not bad_list_path.exists():
            return set()

        bad_paths: set[Path] = set()
        for raw_line in bad_list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = self.raw_root / path
            bad_paths.add(path)
        return bad_paths

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
        from decord import VideoReader, cpu

        if video_path in self._bad_video_paths:
            raise RuntimeError(f"Video is marked unreadable: {video_path}")

        reader = self._readers.get(video_path)
        if reader is None:
            try:
                reader = VideoReader(str(video_path), ctx=cpu(0))
            except Exception as exc:
                raise RuntimeError(f"Could not open video: {video_path}") from exc
            self._readers[video_path] = reader
            while len(self._readers) > self._max_open_readers:
                self._readers.popitem(last=False)
            return reader

        self._readers.move_to_end(video_path)
        return reader

    def _release_reader(self, video_path: Path) -> None:
        self._readers.pop(video_path, None)

    def _mark_bad_video(self, video_path: Path) -> None:
        self._bad_video_paths.add(video_path)
        self._release_reader(video_path)

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

    def _apply_color_jitter(self, patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.color_jitter_prob <= 0 or rng.random() >= self.color_jitter_prob:
            return patch

        image = patch.astype(np.float32)
        if self.brightness_jitter > 0:
            image *= 1.0 + rng.uniform(-self.brightness_jitter, self.brightness_jitter)

        if self.contrast_jitter > 0:
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = (image - mean) * (1.0 + rng.uniform(-self.contrast_jitter, self.contrast_jitter)) + mean

        image = np.clip(image, 0, 255).astype(np.uint8)
        if self.saturation_jitter > 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= 1.0 + rng.uniform(-self.saturation_jitter, self.saturation_jitter)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image

    def _apply_patch_augmentations(self, patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        patch = self._apply_color_jitter(patch, rng)

        if self.grayscale_prob > 0 and rng.random() < self.grayscale_prob:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            patch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if self.blur_prob > 0 and rng.random() < self.blur_prob:
            kernel_size = int(rng.choice(np.asarray([3, 5], dtype=np.int32)))
            patch = cv2.GaussianBlur(patch, (kernel_size, kernel_size), 0)

        if self.noise_prob > 0 and rng.random() < self.noise_prob:
            noise = rng.normal(loc=0.0, scale=255.0 * 0.02, size=patch.shape)
            patch = np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(patch)

    def _trim_and_resize_template(self, patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.template_trim_jitter <= 0:
            return patch

        height, width = patch.shape[:2]
        trim_fraction = rng.uniform(0.0, self.template_trim_jitter)
        crop_scale = max(0.5, 1.0 - trim_fraction)
        crop_w = max(2, int(round(width * crop_scale)))
        crop_h = max(2, int(round(height * crop_scale)))
        x1 = max(0, (width - crop_w) // 2)
        y1 = max(0, (height - crop_h) // 2)
        cropped = patch[y1 : y1 + crop_h, x1 : x1 + crop_w]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def _maybe_flip_pair(
        self,
        template_patch: np.ndarray,
        search_patch: np.ndarray,
        bbox_xyxy: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.horizontal_flip_prob <= 0 or rng.random() >= self.horizontal_flip_prob:
            return template_patch, search_patch, bbox_xyxy

        flipped_bbox = bbox_xyxy.copy()
        old_x1 = float(bbox_xyxy[0])
        old_x2 = float(bbox_xyxy[2])
        flipped_bbox[0] = self.search_size - old_x2
        flipped_bbox[2] = self.search_size - old_x1
        flipped_bbox[[0, 2]] = np.clip(flipped_bbox[[0, 2]], 0.0, float(self.search_size))
        return (
            np.ascontiguousarray(template_patch[:, ::-1]),
            np.ascontiguousarray(search_patch[:, ::-1]),
            flipped_bbox.astype(np.float32),
        )

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
                # Some competition videos are unreadable/corrupt on disk; resample
                # instead of crashing the whole epoch.
                self._mark_bad_video(sequence.video_path)
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
                    area_scale=self.search_area_scale * scale_factor,
                )
                search_bbox_xywh = project_box_to_crop(
                    search_box_xywh=search_box,
                    reference_box_xywh=template_box,
                    out_size=self.search_size,
                    context_amount=self.context_amount,
                    center_override=jittered_center,
                    area_scale=self.search_area_scale * scale_factor,
                )
                x, y, w, h = [float(v) for v in search_bbox_xywh]
                bbox = np.asarray([x, y, x + w, y + h], dtype=np.float32)
                template_patch, search_patch, bbox = self._maybe_flip_pair(template_patch, search_patch, bbox, rng)
                template_patch = self._trim_and_resize_template(template_patch, rng)
                template_patch = self._apply_patch_augmentations(template_patch, rng)
                search_patch = self._apply_patch_augmentations(search_patch, rng)
                labelcls2, labelxff, _, labelcls3, weightxff = self.anchor_target.get(bbox, self.output_size)
            except ValueError as exc:
                last_error = exc
                continue

            return {
                "template": frame_to_tensor(template_patch),
                "search": frame_to_tensor(search_patch),
                "bbox": torch.from_numpy(bbox.copy()),
                "label_cls2": torch.from_numpy(labelcls2.copy()),
                "labelxff": torch.from_numpy(labelxff.copy()),
                "labelcls3": torch.from_numpy(labelcls3.copy()),
                "weightxff": torch.from_numpy(weightxff.copy()),
                "seq_id": sequence.seq_id,
                "template_index": template_index,
                "search_index": search_index,
            }

        if last_error is not None:
            raise RuntimeError(f"Could not sample a valid training pair after {max_attempts} attempts.") from last_error
        raise RuntimeError("Could not sample a valid training pair.")
