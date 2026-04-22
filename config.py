from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ModelSettings:
    backbone: str = "small"
    feature_channels: int = 96
    pretrained: bool = True
    normalize_input: bool = True
    template_size: int = 127
    search_size: int = 255
    context_amount: float = 0.5


@dataclass(slots=True)
class TrainSettings:
    dataset_root: str = "data/raw/UAV123"
    processed_root: str = "data/processed"
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 0.0001
    backbone_learning_rate: float = 0.00001
    weight_decay: float = 0.0001
    freeze_backbone: bool = False
    unfreeze_last_n_backbone_blocks: int = 0
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    cls_weight: float = 1.0
    reg_weight: float = 5.0
    smooth_l1_beta: float = 1.0
    num_workers: int = 2
    pin_memory: bool = True
    train_samples_per_epoch: int = 512
    val_samples_per_epoch: int = 128


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


def load_config(config_path=None):
    path = Path(config_path) if config_path else Path(__file__).with_name("config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Project config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level project config must be a mapping")

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
    if train.backbone_learning_rate <= 0:
        raise ValueError("train.backbone_learning_rate must be positive")
    if train.weight_decay < 0:
        raise ValueError("train.weight_decay cannot be negative")
    if train.unfreeze_last_n_backbone_blocks < 0:
        raise ValueError("train.unfreeze_last_n_backbone_blocks cannot be negative")
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
