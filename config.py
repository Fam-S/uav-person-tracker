from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ModelSettings:
    backbone: str = "mobileone_s2"
    feature_channels: int = 192
    pretrained: bool = False
    pretrained_path: str | None = None
    normalize_input: bool = True
    template_size: int = 127
    search_size: int = 255
    context_amount: float = 0.5


@dataclass(slots=True)
class TrainSettings:
    dataset_root: str = "data/raw"
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    smooth_l1_beta: float = 1.0
    num_workers: int = 2
    pin_memory: bool = True
    train_samples_per_epoch: int = 2048
    translation_jitter: float = 0.15
    scale_jitter: float = 0.1


@dataclass(slots=True)
class InferSettings:
    checkpoint: str = "checkpoints/best.pth"
    video_path: str = "data/input/sample2.mp4"
    output_path: str = "data/output/tracked_output.mp4"
    confidence_threshold: float = 0.35
    device: str = "cpu"
    display_output: bool = True
    save_output: bool = True
    debug: bool = True


@dataclass(slots=True)
class TrackingSettings:
    backend: str = "csrt"
    checkpoint: str | None = None
    uncertain_confidence_threshold: float = 0.55
    lost_confidence_threshold: float = 0.35
    trail_length: int = 20
    search_radius_scale: float = 2.0
    selection_aspect_ratio: float = 0.5
    default_selection_height_fraction: float = 0.2
    template_crop_scale: float = 1.0
    search_crop_scale: float = 2.0
    track_max_width: int = 384


@dataclass(slots=True)
class ProjectConfig:
    model: ModelSettings
    train: TrainSettings
    infer: InferSettings
    tracking: TrackingSettings
    config_path: Path


def validate_tracking_settings(tracking):
    if tracking.trail_length < 0:
        raise ValueError("tracking.trail_length cannot be negative")
    if tracking.search_radius_scale < 1.0:
        raise ValueError("tracking.search_radius_scale must be >= 1.0")
    if tracking.selection_aspect_ratio <= 0:
        raise ValueError("tracking.selection_aspect_ratio must be positive")
    if not 0 < tracking.default_selection_height_fraction < 1:
        raise ValueError("tracking.default_selection_height_fraction must be between 0 and 1")
    if tracking.template_crop_scale <= 0:
        raise ValueError("tracking.template_crop_scale must be positive")
    if tracking.search_crop_scale < 1.0:
        raise ValueError("tracking.search_crop_scale must be >= 1.0")
    if tracking.track_max_width <= 0:
        raise ValueError("tracking.track_max_width must be positive")


def _read_section(data, section):
    value = data.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return value


def _apply_overrides(raw: dict, overrides: dict[str, str]) -> None:
    for key, value in overrides.items():
        parts = key.split(".")
        section_name = parts[0]
        field_name = ".".join(parts[1:])
        if section_name not in raw:
            raw[section_name] = {}
        raw[section_name][field_name] = _parse_value(value)


def _parse_value(value: str):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(config_path=None, overrides: dict[str, str] | None = None):
    path = Path(config_path) if config_path else Path(__file__).with_name("config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Project config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level project config must be a mapping")

    if overrides:
        _apply_overrides(raw, overrides)

    model = ModelSettings(**_read_section(raw, "model"))
    train = TrainSettings(**_read_section(raw, "train"))
    infer = InferSettings(**_read_section(raw, "infer"))
    tracking = TrackingSettings(**_read_section(raw, "tracking"))

    if model.template_size <= 0 or model.search_size <= 0:
        raise ValueError("model template/search sizes must be positive")
    if train.batch_size <= 0:
        raise ValueError("train.batch_size must be positive")
    if train.epochs <= 0:
        raise ValueError("train.epochs must be positive")
    if train.learning_rate <= 0:
        raise ValueError("train.learning_rate must be positive")
    if train.weight_decay < 0:
        raise ValueError("train.weight_decay cannot be negative")
    if infer.confidence_threshold < 0 or infer.confidence_threshold > 1:
        raise ValueError("infer.confidence_threshold must be between 0 and 1")
    validate_tracking_settings(tracking)

    return ProjectConfig(
        model=model,
        train=train,
        infer=infer,
        tracking=tracking,
        config_path=path,
    )
