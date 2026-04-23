from __future__ import annotations

from pathlib import Path

import pytest

from config import get_config_value, list_config_keys, load_config, load_raw_config, set_config_value
from train import config_cli


def _write_config(config_path: Path) -> None:
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
                "  dataset_root: data/raw",
                "  batch_size: 1",
                "  epochs: 1",
                "  learning_rate: 0.0001",
                "  weight_decay: 0.0",
                "  device: cpu",
                "  checkpoint_dir: checkpoints",
                "  smooth_l1_beta: 1.0",
                "  num_workers: 0",
                "  pin_memory: false",
                "  train_samples_per_epoch: 1",
                "infer:",
                "  checkpoint: checkpoints/best.pth",
                "  video_path: data/input/sample2.mp4",
                "  output_path: data/output/tracked_output.mp4",
                "  confidence_threshold: 0.35",
                "  device: cpu",
                "  display_output: true",
                "  save_output: true",
                "  debug: true",
                "tracking:",
                "  backend: siamapn",
                "  checkpoint: checkpoints/best.pth",
                "  uncertain_confidence_threshold: 0.55",
                "  lost_confidence_threshold: 0.35",
                "  trail_length: 20",
                "  search_radius_scale: 2.0",
                "  selection_aspect_ratio: 0.5",
                "  default_selection_height_fraction: 0.2",
                "  template_crop_scale: 1.0",
                "  search_crop_scale: 2.0",
                "  track_max_width: 384",
            ]
        ),
        encoding="utf-8",
    )


def test_config_helpers_support_dotted_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    raw, _ = load_raw_config(config_path)

    assert get_config_value(raw, "train.epochs") == 1

    set_config_value(raw, "train.epochs", 3)
    set_config_value(raw, "tracking.backend", "csrt")

    assert get_config_value(raw, "train.epochs") == 3
    assert get_config_value(raw, "tracking.backend") == "csrt"
    assert "train.epochs" in list_config_keys(raw)
    assert "tracking.backend" in list_config_keys(raw)


def test_config_cli_set_updates_file_with_typed_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        ["project-config", "--config", str(config_path), "set", "train.epochs", "5"],
    )

    config_cli.main()

    config = load_config(config_path)
    captured = capsys.readouterr()

    assert config.train.epochs == 5
    assert f"updated train.epochs in {config_path}" in captured.out


def test_config_cli_list_keys_prints_dotted_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        ["project-config", "--config", str(config_path), "list-keys"],
    )

    config_cli.main()

    captured = capsys.readouterr()
    assert "train.epochs" in captured.out
    assert "tracking.backend" in captured.out
