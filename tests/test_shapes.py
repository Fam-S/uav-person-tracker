from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

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
        search_size=287,
        samples_per_epoch=1,
        seed=5,
    )

    sample = dataset[0]
    assert set(sample) == {
        "template",
        "search",
        "bbox",
        "label_cls2",
        "labelxff",
        "labelcls3",
        "weightxff",
        "seq_id",
        "template_index",
        "search_index",
    }
    assert sample["template"].shape == (3, 127, 127)
    assert sample["search"].shape == (3, 287, 287)
    assert sample["template"].dtype == torch.float32
    assert sample["search"].dtype == torch.float32
    assert sample["bbox"].shape == (4,)
    assert sample["label_cls2"].shape == (1, 21, 21)
    assert sample["labelxff"].shape == (4, 21, 21)
    assert sample["labelcls3"].shape == (1, 21, 21)
    assert sample["weightxff"].shape == (1, 21, 21)
    assert torch.isfinite(sample["bbox"]).all()


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

    dataset = CompetitionSiameseDataset(
        raw_root=raw_root,
        template_size=127,
        search_size=255,
        samples_per_epoch=1,
        seed=0,
    )
    real_load_frame_pair = dataset._load_frame_pair

    def fake_load_frame_pair(video_path: Path, template_index: int, search_index: int) -> tuple[np.ndarray, np.ndarray]:
        if video_path == broken_video_path:
            raise RuntimeError(f"Could not open video: {video_path}")
        return real_load_frame_pair(video_path, template_index, search_index)

    monkeypatch.setattr(dataset, "_load_frame_pair", fake_load_frame_pair)

    broken_sequence = dataset.indexed_sequences[0].sequence
    good_sequence = dataset.indexed_sequences[1].sequence
    sampled_pairs = iter([(broken_sequence, 0, 1), (good_sequence, 0, 1)])

    def fake_sample_pair(rng):
        _ = rng
        return next(sampled_pairs)

    monkeypatch.setattr(dataset, "_sample_pair", fake_sample_pair)

    sample = dataset[0]

    assert sample["seq_id"] == "dataset1/good"


def test_dataloader_smoke_with_workers(tmp_path: Path):
    raw_root = _make_raw_root(tmp_path)
    dataset = CompetitionSiameseDataset(
        raw_root=raw_root,
        template_size=127,
        search_size=287,
        samples_per_epoch=4,
        seed=7,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
    )

    batch = next(iter(dataloader))

    assert batch["template"].shape == (2, 3, 127, 127)
    assert batch["search"].shape == (2, 3, 287, 287)
    assert batch["bbox"].shape == (2, 4)
    assert batch["label_cls2"].shape == (2, 1, 21, 21)
    assert batch["labelxff"].shape == (2, 4, 21, 21)
    assert batch["labelcls3"].shape == (2, 1, 21, 21)
    assert batch["weightxff"].shape == (2, 1, 21, 21)
    assert batch["template"].dtype == torch.float32
    assert batch["search"].dtype == torch.float32
    assert torch.isfinite(batch["bbox"]).all()
    assert list(batch["seq_id"]) == ["dataset1/person1", "dataset1/person1"]
    assert batch["template_index"].shape == (2,)
    assert batch["search_index"].shape == (2,)
