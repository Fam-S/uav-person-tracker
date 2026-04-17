from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


ATTRIBUTE_NAMES = (
    "IV",
    "SV",
    "POC",
    "FOC",
    "OV",
    "FM",
    "CM",
    "BC",
    "SOB",
    "ARC",
    "VC",
    "LR",
)
CONFIG_ENTRY_PATTERN = re.compile(
    r"struct\('name','(?P<name>[^']+)'"
    r".*?'path','(?P<path>[^']+)'"
    r".*?'startFrame',(?P<start>\d+)"
    r".*?'endFrame',(?P<end>\d+)"
    r".*?'nz',(?P<nz>\d+)"
    r".*?'ext','(?P<ext>[^']+)'",
    re.IGNORECASE,
)
BASE_SEQUENCE_PATTERN = re.compile(r"^(?P<base>.+?)_(?P<segment>\d+)$")


@dataclass(frozen=True)
class SequenceFrame:
    frame_index: int
    image_path: Path
    bbox: tuple[float, float, float, float]
    valid: bool
    invalid_reason: str | None
    area: float
    visibility: str
    image_size: tuple[int, int]


@dataclass(frozen=True)
class UAV123Sequence:
    name: str
    base_sequence: str
    source_name: str
    category: str
    image_dir: Path
    annotation_path: Path
    start_frame: int
    end_frame: int
    frames: tuple[SequenceFrame, ...]
    valid_frames: tuple[SequenceFrame, ...]
    invalid_frames: tuple[SequenceFrame, ...]
    difficulty_tags: dict[str, bool]
    fps: float | None = None
    split_group: str | None = None

    @property
    def valid_frame_count(self) -> int:
        return len(self.valid_frames)


@dataclass(frozen=True)
class UAV123ConfigEntry:
    name: str
    source_name: str
    start_frame: int
    end_frame: int
    zero_padding: int
    extension: str


@dataclass(frozen=True)
class FrameParseResult:
    bbox: tuple[float, float, float, float]
    valid: bool
    invalid_reason: str | None
    area: float
    visibility: str


@dataclass(frozen=True)
class ProcessedDatasetPaths:
    root: Path
    manifest: Path
    annotations_cache: Path
    train_split: Path
    val_split: Path
    summary: Path


@dataclass(frozen=True)
class CropResult:
    image: Image.Image
    bbox_xywh: tuple[float, float, float, float]
    crop_box_xyxy: tuple[float, float, float, float]
    resize_scale: float


@dataclass(frozen=True)
class SiameseSample:
    template: torch.Tensor
    search: torch.Tensor
    search_bbox: torch.Tensor
    template_bbox: torch.Tensor
    sequence_name: str
    template_frame_index: int
    search_frame_index: int
    difficulty_tags: dict[str, bool]


EXPECTED_ATTRIBUTE_ORDER_DESCRIPTION = ", ".join(ATTRIBUTE_NAMES)


def derive_base_sequence(sequence_name: str) -> str:
    match = BASE_SEQUENCE_PATTERN.match(sequence_name)
    if match:
        return match.group("base")
    return sequence_name


def infer_category(sequence_name: str) -> str:
    base = derive_base_sequence(sequence_name)
    match = re.match(r"^[A-Za-z]+", base)
    return match.group(0).lower() if match else "unknown"


def parse_uav123_config(config_path: str | Path) -> dict[str, UAV123ConfigEntry]:
    config_text = Path(config_path).read_text(encoding="utf-8", errors="ignore")
    entries: dict[str, UAV123ConfigEntry] = {}

    for match in CONFIG_ENTRY_PATTERN.finditer(config_text):
        name = match.group("name")
        raw_path = match.group("path").replace("\\", "/").rstrip("/")
        source_name = Path(raw_path).name
        entries[name] = UAV123ConfigEntry(
            name=name,
            source_name=source_name,
            start_frame=int(match.group("start")),
            end_frame=int(match.group("end")),
            zero_padding=int(match.group("nz")),
            extension=match.group("ext"),
        )

    if not entries:
        raise ValueError(f"No UAV123 sequence entries found in {config_path}")

    return entries


def parse_attribute_file(attribute_path: Path) -> dict[str, bool]:
    if not attribute_path.exists():
        return {name: False for name in ATTRIBUTE_NAMES}

    raw_values = [
        value.strip()
        for value in attribute_path.read_text(encoding="utf-8", errors="ignore").split(",")
        if value.strip()
    ]
    if len(raw_values) != len(ATTRIBUTE_NAMES):
        return {name: False for name in ATTRIBUTE_NAMES}

    return {name: bool(int(value)) for name, value in zip(ATTRIBUTE_NAMES, raw_values)}


def parse_uav123_bbox(raw_value: str) -> FrameParseResult:
    text = raw_value.strip()
    if not text:
        return FrameParseResult((0.0, 0.0, 0.0, 0.0), False, "empty_row", 0.0, "absent")

    parts = [part.strip() for part in text.replace("\t", ",").split(",") if part.strip()]
    if len(parts) != 4:
        return FrameParseResult((0.0, 0.0, 0.0, 0.0), False, "malformed_row", 0.0, "absent")

    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return FrameParseResult((0.0, 0.0, 0.0, 0.0), False, "non_numeric", 0.0, "absent")

    if any(math.isnan(value) for value in values):
        return FrameParseResult((0.0, 0.0, 0.0, 0.0), False, "nan_bbox", 0.0, "absent")

    x, y, w, h = values
    area = max(w, 0.0) * max(h, 0.0)
    if w <= 0 or h <= 0:
        return FrameParseResult((x, y, w, h), False, "non_positive_size", area, "absent")

    return FrameParseResult((x, y, w, h), True, None, area, "visible")


def validate_frame_bbox(frame_info: FrameParseResult, image_size: tuple[int, int]) -> FrameParseResult:
    if not frame_info.valid:
        return frame_info

    image_width, image_height = image_size
    x, y, w, h = frame_info.bbox
    x2 = x + w
    y2 = y + h

    inter_x1 = max(0.0, x)
    inter_y1 = max(0.0, y)
    inter_x2 = min(float(image_width), x2)
    inter_y2 = min(float(image_height), y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersect_area = inter_w * inter_h
    if intersect_area <= 0.0:
        return FrameParseResult(frame_info.bbox, False, "outside_image", frame_info.area, "absent")

    visibility = "visible"
    if intersect_area < frame_info.area:
        visibility = "clipped"
    if intersect_area < 400.0:
        visibility = "low_resolution" if visibility == "visible" else "clipped_low_resolution"

    return FrameParseResult(frame_info.bbox, True, None, frame_info.area, visibility)


class UAV123DatasetLoader:
    def __init__(
        self,
        dataset_root: str | Path,
        subset: str = "UAV123",
        config_filename: str = "configSeqs.m",
        processed_root: str | Path | None = None,
        split: str | None = None,
        min_valid_frames: int = 2,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.subset = subset
        self.min_valid_frames = min_valid_frames
        self.annotation_root = self.dataset_root / "anno" / subset
        self.frame_root = self.dataset_root / "data_seq" / subset
        self.attribute_root = self.annotation_root / "att"
        self.config_path = self.dataset_root / config_filename
        self.processed_root = Path(processed_root) if processed_root is not None else None
        self.split = split

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        if not self.annotation_root.exists():
            raise FileNotFoundError(f"Annotation directory does not exist: {self.annotation_root}")
        if not self.frame_root.exists():
            raise FileNotFoundError(f"Frame directory does not exist: {self.frame_root}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {self.config_path}")

        self.sequence_configs = parse_uav123_config(self.config_path)

    def load_sequences(self) -> list[UAV123Sequence]:
        if self.processed_root is not None:
            paths = get_processed_dataset_paths(self.processed_root)
            if paths.manifest.exists() and paths.annotations_cache.exists():
                return self.load_sequences_from_processed(paths)

        return self.load_sequences_from_raw()

    def load_sequences_from_raw(self) -> list[UAV123Sequence]:
        sequences: list[UAV123Sequence] = []

        for annotation_path in sorted(self.annotation_root.glob("*.txt")):
            sequence_name = annotation_path.stem
            if sequence_name == "att":
                continue

            config_entry = self.sequence_configs.get(sequence_name)
            if config_entry is None:
                continue

            sequence = self._build_sequence_from_raw(annotation_path, config_entry)
            if sequence.valid_frame_count >= self.min_valid_frames:
                sequences.append(sequence)

        if not sequences:
            raise ValueError(f"No valid sequences were loaded from {self.annotation_root}.")

        return sequences

    def _build_sequence_from_raw(self, annotation_path: Path, config_entry: UAV123ConfigEntry) -> UAV123Sequence:
        image_dir = self.frame_root / config_entry.source_name
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory for sequence {config_entry.name} does not exist: {image_dir}")

        raw_annotations = annotation_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        frame_indices = range(config_entry.start_frame, config_entry.end_frame + 1)
        expected_length = config_entry.end_frame - config_entry.start_frame + 1
        if len(raw_annotations) != expected_length:
            raise ValueError(
                f"Annotation length mismatch for {config_entry.name}: expected {expected_length}, got {len(raw_annotations)}"
            )

        frames: list[SequenceFrame] = []
        for frame_index, raw_bbox in zip(frame_indices, raw_annotations):
            frame_info = parse_uav123_bbox(raw_bbox)
            image_path = image_dir / f"{frame_index:0{config_entry.zero_padding}d}.{config_entry.extension}"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing frame image for sequence {config_entry.name}: {image_path}")

            with Image.open(image_path) as image:
                image_size = image.size
            frame_info = validate_frame_bbox(frame_info, image_size=image_size)

            frames.append(
                SequenceFrame(
                    frame_index=frame_index,
                    image_path=image_path,
                    bbox=frame_info.bbox,
                    valid=frame_info.valid,
                    invalid_reason=frame_info.invalid_reason,
                    area=frame_info.area,
                    visibility=frame_info.visibility,
                    image_size=image_size,
                )
            )

        valid_frames = tuple(frame for frame in frames if frame.valid)
        invalid_frames = tuple(frame for frame in frames if not frame.valid)
        base_sequence = derive_base_sequence(config_entry.name)
        difficulty_tags = parse_attribute_file(self.attribute_root / f"{config_entry.name}.txt")

        return UAV123Sequence(
            name=config_entry.name,
            base_sequence=base_sequence,
            source_name=config_entry.source_name,
            category=infer_category(config_entry.name),
            image_dir=image_dir,
            annotation_path=annotation_path,
            start_frame=config_entry.start_frame,
            end_frame=config_entry.end_frame,
            frames=tuple(frames),
            valid_frames=valid_frames,
            invalid_frames=invalid_frames,
            difficulty_tags=difficulty_tags,
            fps=None,
            split_group=base_sequence,
        )

    def load_sequences_from_processed(self, paths: ProcessedDatasetPaths) -> list[UAV123Sequence]:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        annotations_cache = json.loads(paths.annotations_cache.read_text(encoding="utf-8"))
        allowed_names = set(self._read_split_names(paths)) if self.split else None

        sequences: list[UAV123Sequence] = []
        for record in manifest["sequences"]:
            sequence_name = record["sequence_name"]
            if allowed_names is not None and sequence_name not in allowed_names:
                continue

            frame_records = annotations_cache[sequence_name]["frames"]
            frames = tuple(
                SequenceFrame(
                    frame_index=frame_record["frame_index"],
                    image_path=Path(frame_record["image_path"]),
                    bbox=tuple(frame_record["bbox"]),
                    valid=frame_record["valid"],
                    invalid_reason=frame_record["invalid_reason"],
                    area=frame_record["area"],
                    visibility=frame_record["visibility"],
                    image_size=tuple(frame_record.get("image_size", [0, 0])),
                )
                for frame_record in frame_records
            )
            valid_frames = tuple(frame for frame in frames if frame.valid)
            invalid_frames = tuple(frame for frame in frames if not frame.valid)

            if len(valid_frames) < self.min_valid_frames:
                continue

            sequences.append(
                UAV123Sequence(
                    name=sequence_name,
                    base_sequence=record["base_sequence"],
                    source_name=record["source_name"],
                    category=record["category"],
                    image_dir=Path(record["frames_dir"]),
                    annotation_path=Path(record["annotation_file"]),
                    start_frame=record["start_frame"],
                    end_frame=record["end_frame"],
                    frames=frames,
                    valid_frames=valid_frames,
                    invalid_frames=invalid_frames,
                    difficulty_tags=record["difficulty_tags"],
                    fps=record.get("fps"),
                    split_group=record.get("split_group"),
                )
            )

        if not sequences:
            raise ValueError("No valid sequences were loaded from processed metadata.")
        return sequences

    def _read_split_names(self, paths: ProcessedDatasetPaths) -> list[str]:
        if self.split == "train":
            split_path = paths.train_split
        elif self.split in {"val", "validation"}:
            split_path = paths.val_split
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        if not split_path.exists():
            raise FileNotFoundError(f"Split file does not exist: {split_path}")

        return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_processed_dataset_paths(processed_root: str | Path) -> ProcessedDatasetPaths:
    root = Path(processed_root)
    return ProcessedDatasetPaths(
        root=root,
        manifest=root / "manifest.json",
        annotations_cache=root / "annotations_cache.json",
        train_split=root / "splits" / "train.txt",
        val_split=root / "splits" / "val.txt",
        summary=root / "preprocess_summary.json",
    )


def xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def crop_square_patch(
    image: Image.Image,
    bbox_xywh: tuple[float, float, float, float],
    output_size: int,
    context_amount: float,
    scale_multiplier: float = 1.0,
    rng: random.Random | None = None,
    translation_jitter: float = 0.0,
    scale_jitter: float = 0.0,
) -> CropResult:
    x, y, w, h = bbox_xywh
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    context = context_amount * (w + h)
    base_crop_side = math.sqrt((w + context) * (h + context)) * scale_multiplier
    crop_side = max(base_crop_side, 1.0)

    if rng is not None and scale_jitter > 0.0:
        crop_side *= 1.0 + rng.uniform(-scale_jitter, scale_jitter)
        crop_side = max(crop_side, 1.0)

    if rng is not None and translation_jitter > 0.0:
        max_offset = translation_jitter * crop_side
        center_x += rng.uniform(-max_offset, max_offset)
        center_y += rng.uniform(-max_offset, max_offset)

    half_side = crop_side / 2.0
    left = center_x - half_side
    top = center_y - half_side
    right = center_x + half_side
    bottom = center_y + half_side

    crop = image.crop((left, top, right, bottom)).resize((output_size, output_size), resample=Image.BILINEAR)
    scale = output_size / crop_side
    x1, y1, x2, y2 = xywh_to_xyxy(bbox_xywh)
    mapped_x1 = (x1 - left) * scale
    mapped_y1 = (y1 - top) * scale
    mapped_x2 = (x2 - left) * scale
    mapped_y2 = (y2 - top) * scale

    clipped_x1 = min(max(mapped_x1, 0.0), float(output_size))
    clipped_y1 = min(max(mapped_y1, 0.0), float(output_size))
    clipped_x2 = min(max(mapped_x2, 0.0), float(output_size))
    clipped_y2 = min(max(mapped_y2, 0.0), float(output_size))

    return CropResult(
        image=crop,
        bbox_xywh=(
            clipped_x1,
            clipped_y1,
            max(clipped_x2 - clipped_x1, 0.0),
            max(clipped_y2 - clipped_y1, 0.0),
        ),
        crop_box_xyxy=(left, top, right, bottom),
        resize_scale=scale,
    )


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor / 255.0


def summarize_sequences(sequences: Iterable[UAV123Sequence]) -> dict[str, int]:
    sequence_list = list(sequences)
    return {
        "sequence_count": len(sequence_list),
        "valid_frame_count": sum(sequence.valid_frame_count for sequence in sequence_list),
        "invalid_frame_count": sum(len(sequence.invalid_frames) for sequence in sequence_list),
    }


def serialize_sequence_manifest(sequence: UAV123Sequence) -> dict[str, Any]:
    return {
        "sequence_name": sequence.name,
        "base_sequence": sequence.base_sequence,
        "category": sequence.category,
        "frames_dir": str(sequence.image_dir),
        "annotation_file": str(sequence.annotation_path),
        "source_name": sequence.source_name,
        "num_frames": len(sequence.frames),
        "num_annotations": len(sequence.frames),
        "num_valid_frames": len(sequence.valid_frames),
        "num_invalid_frames": len(sequence.invalid_frames),
        "fps": sequence.fps,
        "split_group": sequence.split_group,
        "start_frame": sequence.start_frame,
        "end_frame": sequence.end_frame,
        "difficulty_tags": sequence.difficulty_tags,
    }


def serialize_sequence_annotations(sequence: UAV123Sequence) -> dict[str, Any]:
    return {
        "sequence_name": sequence.name,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "image_path": str(frame.image_path),
                "bbox": list(frame.bbox),
                "valid": frame.valid,
                "invalid_reason": frame.invalid_reason,
                "area": frame.area,
                "visibility": frame.visibility,
                "image_size": list(frame.image_size),
            }
            for frame in sequence.frames
        ],
    }


# ─── Pair Sampler ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SiameseTrainingPair:
    sequence_name: str
    difficulty_tags: dict[str, bool]
    template: SequenceFrame
    search: SequenceFrame
    frame_gap: int


class UAV123PairSampler:
    def __init__(
        self,
        sequences: list[UAV123Sequence],
        max_frame_gap: int = 100,
    ) -> None:
        if not sequences:
            raise ValueError("Pair sampler requires at least one sequence.")
        if max_frame_gap < 1:
            raise ValueError("max_frame_gap must be at least 1.")

        self.sequences = [sequence for sequence in sequences if sequence.valid_frame_count >= 2]
        if not self.sequences:
            raise ValueError("Pair sampler requires sequences with at least two valid frames.")

        self.max_frame_gap = max_frame_gap

    def sample_pair(self, rng: random.Random | None = None) -> SiameseTrainingPair:
        rng = rng or random.Random()
        sequence = rng.choice(self.sequences)
        return self.sample_pair_from_sequence(sequence, rng=rng)

    def sample_pair_from_sequence(
        self,
        sequence: UAV123Sequence,
        rng: random.Random | None = None,
    ) -> SiameseTrainingPair:
        if sequence.valid_frame_count < 2:
            raise ValueError(f"Sequence {sequence.name} does not have enough valid frames.")

        rng = rng or random.Random()
        candidate_pairs: list[tuple[int, int]] = []
        for template_index in range(sequence.valid_frame_count - 1):
            template_frame = sequence.valid_frames[template_index]
            for search_index in range(template_index + 1, sequence.valid_frame_count):
                search_frame = sequence.valid_frames[search_index]
                gap = search_frame.frame_index - template_frame.frame_index
                if gap > self.max_frame_gap:
                    break
                candidate_pairs.append((template_index, search_index))

        if not candidate_pairs:
            raise ValueError(
                f"Sequence {sequence.name} does not contain valid frame pairs within max_frame_gap={self.max_frame_gap}."
            )

        template_index, search_index = rng.choice(candidate_pairs)
        template = sequence.valid_frames[template_index]
        search = sequence.valid_frames[search_index]
        return SiameseTrainingPair(
            sequence_name=sequence.name,
            difficulty_tags=sequence.difficulty_tags,
            template=template,
            search=search,
            frame_gap=search.frame_index - template.frame_index,
        )


# ─── Siamese Dataset ─────────────────────────────────────────────────────────

class UAV123SiameseDataset(Dataset[SiameseSample]):
    sampling_policy = "deterministic_per_index"

    def __init__(
        self,
        dataset_root: str | Path,
        subset: str = "UAV123",
        config_filename: str = "configSeqs.m",
        processed_root: str | Path | None = None,
        split: str | None = None,
        template_size: int = 127,
        search_size: int = 255,
        context_amount: float = 0.5,
        search_scale_multiplier: float = 2.0,
        search_translation_jitter: float = 0.15,
        search_scale_jitter: float = 0.1,
        max_frame_gap: int = 100,
        samples_per_epoch: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.template_size = template_size
        self.search_size = search_size
        self.context_amount = context_amount
        self.search_scale_multiplier = search_scale_multiplier
        self.search_translation_jitter = search_translation_jitter
        self.search_scale_jitter = search_scale_jitter
        self.base_seed = 0 if seed is None else int(seed)

        self.loader = UAV123DatasetLoader(
            dataset_root=dataset_root,
            subset=subset,
            config_filename=config_filename,
            processed_root=processed_root,
            split=split,
        )
        self.sequences = self.loader.load_sequences()

        self.pair_sampler = UAV123PairSampler(self.sequences, max_frame_gap=max_frame_gap)
        self.samples_per_epoch = samples_per_epoch or len(self.sequences)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng_for_index(self, index: int) -> random.Random:
        return random.Random(self.base_seed + int(index))

    def __getitem__(self, index: int) -> SiameseSample:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")

        rng = self._rng_for_index(index)
        pair = self.pair_sampler.sample_pair(rng=rng)

        with Image.open(pair.template.image_path) as template_source:
            template_image = template_source.convert("RGB")
        with Image.open(pair.search.image_path) as search_source:
            search_image = search_source.convert("RGB")

        template_crop = crop_square_patch(
            image=template_image,
            bbox_xywh=pair.template.bbox,
            output_size=self.template_size,
            context_amount=self.context_amount,
            scale_multiplier=1.0,
            rng=None,
            translation_jitter=0.0,
            scale_jitter=0.0,
        )
        search_crop = crop_square_patch(
            image=search_image,
            bbox_xywh=pair.search.bbox,
            output_size=self.search_size,
            context_amount=self.context_amount,
            scale_multiplier=self.search_scale_multiplier,
            rng=rng,
            translation_jitter=self.search_translation_jitter,
            scale_jitter=self.search_scale_jitter,
        )

        return SiameseSample(
            template=pil_to_tensor(template_crop.image),
            search=pil_to_tensor(search_crop.image),
            search_bbox=torch.tensor(search_crop.bbox_xywh, dtype=torch.float32),
            template_bbox=torch.tensor(template_crop.bbox_xywh, dtype=torch.float32),
            sequence_name=pair.sequence_name,
            template_frame_index=pair.template.frame_index,
            search_frame_index=pair.search.frame_index,
            difficulty_tags=pair.difficulty_tags,
        )
