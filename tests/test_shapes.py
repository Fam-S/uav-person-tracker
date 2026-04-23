from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

import data.competition_siamese_dataset as competition_siamese_dataset
from data import CompetitionSiameseDataset


def _write_video(video_path: Path, num_frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 64),
    )
    if not writer.isOpened():
        raise RuntimeError("Test video writer could not open.")
    for frame_index in range(num_frames):
        frame = torch.full((64, 64, 3), fill_value=40 * (frame_index + 1), dtype=torch.uint8).numpy()
        writer.write(frame)
    writer.release()


def _make_raw_root(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    sequence_dir = raw_root / "dataset1" / "person1"
    metadata_dir = raw_root / "metadata"
    sequence_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    video_path = sequence_dir / "person1.mp4"
    _write_video(video_path)

    annotation_path = sequence_dir / "annotation.txt"
    annotation_path.write_text("10,12,20,24\n12,14,20,24\n14,16,20,24\n", encoding="utf-8")

    manifest = {
        "train": {
            "dataset1/person1": {
                "dataset": "dataset1",
                "seq_name": "person1",
                "video_path": "dataset1/person1/person1.mp4",
                "annotation_path": "dataset1/person1/annotation.txt",
                "n_frames": 3,
                "native_fps": 10,
            }
        },
        "public_lb": {},
    }
    (metadata_dir / "contestant_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw_root


def test_dataset_sample_contract(tmp_path: Path):
    raw_root = _make_raw_root(tmp_path)
    dataset = CompetitionSiameseDataset(
        raw_root=raw_root,
        template_size=127,
        search_size=255,
        samples_per_epoch=1,
        seed=5,
    )

    sample = dataset[0]
    assert set(sample) == {
        "template",
        "search",
        "search_bbox_xywh",
        "seq_id",
        "template_index",
        "search_index",
    }
    assert sample["template"].shape == (3, 127, 127)
    assert sample["search"].shape == (3, 255, 255)
    assert sample["template"].dtype == torch.float32
    assert sample["search"].dtype == torch.float32
    assert sample["search_bbox_xywh"].shape == (4,)
    assert torch.isfinite(sample["search_bbox_xywh"]).all()
    assert torch.all(sample["search_bbox_xywh"] >= 0.0)


def test_dataset_skips_unreadable_video_and_resamples(tmp_path: Path, monkeypatch):
    raw_root = tmp_path / "raw"
    metadata_dir = raw_root / "metadata"
    metadata_dir.mkdir(parents=True)

    seq1_dir = raw_root / "dataset1" / "broken"
    seq1_dir.mkdir(parents=True)
    broken_video_path = seq1_dir / "broken.mp4"
    _write_video(broken_video_path)
    (seq1_dir / "annotation.txt").write_text("10,12,20,24\n12,14,20,24\n14,16,20,24\n", encoding="utf-8")

    seq2_dir = raw_root / "dataset1" / "good"
    seq2_dir.mkdir(parents=True)
    good_video_path = seq2_dir / "good.mp4"
    _write_video(good_video_path)
    (seq2_dir / "annotation.txt").write_text("10,12,20,24\n12,14,20,24\n14,16,20,24\n", encoding="utf-8")

    manifest = {
        "train": {
            "dataset1/broken": {
                "dataset": "dataset1",
                "seq_name": "broken",
                "video_path": "dataset1/broken/broken.mp4",
                "annotation_path": "dataset1/broken/annotation.txt",
                "n_frames": 3,
                "native_fps": 10,
            },
            "dataset1/good": {
                "dataset": "dataset1",
                "seq_name": "good",
                "video_path": "dataset1/good/good.mp4",
                "annotation_path": "dataset1/good/annotation.txt",
                "n_frames": 3,
                "native_fps": 10,
            },
        },
        "public_lb": {},
    }
    (metadata_dir / "contestant_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    real_load_frame = competition_siamese_dataset._load_frame

    def fake_load_frame(video_path: Path, frame_index: int) -> np.ndarray:
        if video_path == broken_video_path:
            raise RuntimeError(f"Could not open video: {video_path}")
        return real_load_frame(video_path, frame_index)

    monkeypatch.setattr(competition_siamese_dataset, "_load_frame", fake_load_frame)

    dataset = CompetitionSiameseDataset(
        raw_root=raw_root,
        template_size=127,
        search_size=255,
        samples_per_epoch=1,
        seed=0,
    )

    broken_sequence = dataset.indexed_sequences[0].sequence
    good_sequence = dataset.indexed_sequences[1].sequence
    sampled_pairs = iter([(broken_sequence, 0, 1), (good_sequence, 0, 1)])

    def fake_sample_pair(rng):
        _ = rng
        return next(sampled_pairs)

    monkeypatch.setattr(dataset, "_sample_pair", fake_sample_pair)

    sample = dataset[0]

    assert sample["seq_id"] == "dataset1/good"
    assert broken_video_path in dataset._bad_video_paths
