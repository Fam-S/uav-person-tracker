from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppSettings:
    title: str = "UAV Person Tracker"
    width: int = 1280
    height: int = 800
    min_width: int = 1024
    min_height: int = 640


@dataclass(slots=True)
class VideoSettings:
    target_fps: int = 30
    keep_aspect_ratio: bool = True


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
    track_max_width: int = 640


@dataclass(slots=True)
class OverlaySettings:
    show_confidence: bool = True
    show_trail: bool = True


@dataclass(slots=True)
class AppConfig:
    app: AppSettings
    video: VideoSettings
    tracking: TrackingSettings
    overlay: OverlaySettings
    config_path: Path


def _read_section(data: dict[str, Any], section: str) -> dict[str, Any]:
    value = data.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return value


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else Path(__file__).with_name("app_config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"App config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level app config must be a mapping")

    app = AppSettings(**_read_section(raw, "app"))
    video = VideoSettings(**_read_section(raw, "video"))
    tracking = TrackingSettings(**_read_section(raw, "tracking"))
    overlay = OverlaySettings(**_read_section(raw, "overlay"))

    if app.width <= 0 or app.height <= 0:
        raise ValueError("Window size must be positive")
    if video.target_fps <= 0:
        raise ValueError("video.target_fps must be positive")
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

    return AppConfig(
        app=app,
        video=video,
        tracking=tracking,
        overlay=overlay,
        config_path=path,
    )
