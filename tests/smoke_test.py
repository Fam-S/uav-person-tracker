from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch

from config import load_config
from train.run import SiameseTrainer


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
        frame = torch.full((64, 64, 3), fill_value=50 * (frame_index + 1), dtype=torch.uint8).numpy()
        writer.write(frame)
    writer.release()


def _make_training_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw_root = tmp_path / "raw"
    sequence_dir = raw_root / "dataset1" / "person1"
    metadata_dir = raw_root / "metadata"
    sequence_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    video_path = sequence_dir / "person1.mp4"
    _write_video(video_path)

    annotation_path = sequence_dir / "annotation.txt"
    annotation_path.write_text("8,10,18,22\n10,12,18,22\n12,14,18,22\n", encoding="utf-8")

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

    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  backbone: mobileone_s2",
                "  feature_channels: 64",
                "  pretrained: false",
                "  normalize_input: true",
                "  template_size: 127",
                "  search_size: 255",
                "  context_amount: 0.5",
                "train:",
                f"  dataset_root: {raw_root.as_posix()}",
                "  batch_size: 1",
                "  epochs: 1",
                "  learning_rate: 0.0001",
                "  weight_decay: 0.0",
                "  device: cpu",
                f"  checkpoint_dir: {checkpoint_dir.as_posix()}",
                "  smooth_l1_beta: 1.0",
                "  num_workers: 0",
                "  pin_memory: false",
                "  train_samples_per_epoch: 1",
            ]
        ),
        encoding="utf-8",
    )
    return raw_root, config_path


def test_one_epoch_smoke(tmp_path: Path):
    _, config_path = _make_training_fixture(tmp_path)
    config = load_config(config_path)
    trainer = SiameseTrainer(config)
    dataloader = trainer.build_dataloader()
    stats = trainer.train_epoch(dataloader, epoch=1)
    checkpoint_path = trainer.save_checkpoint(1, stats)

    assert stats.num_batches == 1
    assert torch.isfinite(torch.tensor(stats.mean_total_loss))
    assert checkpoint_path.exists()
