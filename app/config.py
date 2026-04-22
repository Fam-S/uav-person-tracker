from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from config import TrackingSettings, load_config as load_project_config, validate_tracking_settings


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
class OverlaySettings:
    show_confidence: bool = True
    show_trail: bool = True


@dataclass(slots=True)
class AppConfig:
    app: AppSettings
    video: VideoSettings
    tracking: TrackingSettings
    overlay: OverlaySettings
    app_config_path: Path
    project_config_path: Path


def _read_section(data, section):
    value = data.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return value


def load_config(app_config_path: str | Path | None = None, project_config_path: str | Path | None = None) -> AppConfig:
    app_path = Path(app_config_path) if app_config_path else Path(__file__).with_name("app_config.yaml")
    root_path = Path(project_config_path) if project_config_path else Path(__file__).resolve().parents[1] / "config.yaml"
    if not app_path.exists():
        raise FileNotFoundError(f"App config not found: {app_path}")

    with app_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level app config must be a mapping")

    project_config = load_project_config(root_path)
    app = AppSettings(**_read_section(raw, "app"))
    video = VideoSettings(**_read_section(raw, "video"))
    overlay = OverlaySettings(**_read_section(raw, "overlay"))
    tracking_data = asdict(project_config.tracking)
    tracking_data.update(_read_section(raw, "tracking"))
    tracking = TrackingSettings(**tracking_data)

    if app.width <= 0 or app.height <= 0:
        raise ValueError("Window size must be positive")
    if video.target_fps <= 0:
        raise ValueError("video.target_fps must be positive")
    validate_tracking_settings(tracking)

    return AppConfig(
        app=app,
        video=video,
        tracking=tracking,
        overlay=overlay,
        app_config_path=app_path,
        project_config_path=root_path,
    )
