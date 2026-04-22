from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch

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
