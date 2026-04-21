from __future__ import annotations

from collections import deque
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.config import AppConfig
from app.tracking import BBox, TrackerBackend, TrackingResult


class AppController:
    def __init__(self, config: AppConfig, backend: TrackerBackend) -> None:
        self.config = config
        self.backend = backend
        self.ui = None

        self.state = "Idle"
        self.capture: cv2.VideoCapture | None = None
        self.video_path: str | None = None
        self.current_frame: np.ndarray | None = None
        self.current_bbox: BBox | None = None
        self.selected_bbox: BBox | None = None
        self.trail: deque[tuple[int, int]] = deque(maxlen=config.tracking.trail_length)

        self.last_confidence: float | None = None
        self.last_latency_ms: float | None = None
        self.last_fps: float | None = None
        self._last_command: tuple[str, float] | None = None
        self._smoothed_velocity: np.ndarray | None = None
        self.tick_after_id = None
        self.backend_initialized = False
        self.last_tick_started_at: float | None = None

    def bind_ui(self, ui) -> None:
        self.ui = ui
        self.ui.bind_controller(self)
        self.ui.set_backend_name(self.config.tracking.backend)
        self._sync_ui_state()

    def open_video(self) -> None:
        if self.ui is None:
            return

        video_path = self.ui.ask_video_path()
        if not video_path:
            return

        new_capture = cv2.VideoCapture(video_path)
        if not new_capture.isOpened():
            self.ui.show_error("Open Video", f"Could not open video file:\n{video_path}")
            return

        success, frame = new_capture.read()
        if not success or frame is None:
            new_capture.release()
            self.ui.show_error("Open Video", "Could not read the first frame from the selected video.")
            return

        self._release_capture()
        self.capture = new_capture
        self.video_path = video_path
        self.current_frame = frame
        self.state = "Video Loaded"
        self.selected_bbox = None
        self.current_bbox = None
        self.trail.clear()
        self.last_confidence = None
        self.last_latency_ms = None
        self.last_fps = None
        self.backend.reset()
        self.backend_initialized = False
        self._cancel_tick()

        self.ui.clear_selection_box()
        self.ui.set_video_label(video_path)
        self._render_current_frame()
        self.ui.add_event(f"Loaded video: {Path(video_path).name}")
        self.ui.enable_target_selection()
        self._sync_ui_state()

    def enable_target_selection(self) -> None:
        if self.ui is None:
            return
        if self.current_frame is None:
            self.ui.show_info("Select Target", "Open a video before selecting a target.")
            return
        self.ui.enable_target_selection()

    def on_target_selected(self, bbox: BBox) -> None:
        if self.current_frame is None or self.ui is None:
            return

        self.selected_bbox = bbox
        self.current_bbox = bbox
        self.trail.clear()
        self._append_trail_point(bbox)
        self.state = "Target Selected"
        self.last_confidence = None
        self.last_latency_ms = None
        self.backend.reset()
        self.backend_initialized = False

        self.ui.set_persistent_selection_box(bbox)
        self._render_current_frame()
        self.ui.add_event("Target selected")
        self._sync_ui_state()

    def start_tracking(self) -> None:
        if self.ui is None:
            return
        if self.capture is None or self.current_frame is None:
            self.ui.show_info("Start", "Open a video first.")
            return
        if self.selected_bbox is None:
            self.ui.show_info("Start", "Select a target before starting tracking.")
            return

        if not self.backend_initialized:
            try:
                self.backend.initialize(self.current_frame, self.selected_bbox)
            except Exception as exc:  # pragma: no cover - UI error path
                self.ui.show_error("Tracker Initialization", str(exc))
                return
            self.backend_initialized = True
            self.current_bbox = self.selected_bbox

        self.state = "Tracking"
        self.ui.add_event("Tracking started")
        self._schedule_next_tick(initial=True)
        self._sync_ui_state()

    def pause_tracking(self) -> None:
        if self.state not in {"Tracking", "Lost", "Uncertain"}:
            return
        self._cancel_tick()
        self.state = "Paused"
        if self.ui is not None:
            self.ui.add_event("Tracking paused")
        self._sync_ui_state()

    def reset_tracking(self) -> None:
        if self.capture is None or self.ui is None:
            return

        self._cancel_tick()
        self.backend.reset()
        self.backend_initialized = False
        self.selected_bbox = None
        self.current_bbox = None
        self.trail.clear()
        self.last_confidence = None
        self.last_latency_ms = None
        self.last_fps = None
        self._last_command = None
        self._smoothed_velocity = None

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = self.capture.read()
        if success and frame is not None:
            self.current_frame = frame
            self.state = "Video Loaded"
            self.ui.clear_selection_box()
            self._render_current_frame()
            self.ui.add_event("Tracking reset")
            self.ui.enable_target_selection()
        else:
            self.current_frame = None
            self.state = "Idle"
            self.ui.add_event("Reset failed: could not reload first frame")
        self._sync_ui_state()

    def refresh_current_frame(self) -> None:
        if self.current_frame is not None and self.ui is not None:
            self._render_current_frame()

    def shutdown(self) -> None:
        self._cancel_tick()
        self._release_capture()

    def _tick(self) -> None:
        self.tick_after_id = None
        if self.capture is None or self.current_frame is None:
            return

        tick_started_at = perf_counter()
        success, frame = self.capture.read()
        if not success or frame is None:
            self.state = "Paused"
            if self.ui is not None:
                self.ui.add_event("Reached end of video")
            self._sync_ui_state()
            return

        self.current_frame = frame
        result = self.backend.track(frame)
        self.current_bbox = result.bbox
        self.last_confidence = result.confidence
        self.last_latency_ms = result.latency_ms
        self.state = result.state
        if result.bbox is not None:
            self._append_trail_point(result.bbox)
        self.ui.set_persistent_selection_box(result.bbox)
        self._send_drone_command()

        if self.last_tick_started_at is not None:
            elapsed = tick_started_at - self.last_tick_started_at
            if elapsed > 0:
                self.last_fps = 1.0 / elapsed
        self.last_tick_started_at = tick_started_at

        self._render_current_frame()
        self._schedule_next_tick()
        self._sync_ui_state()

    def _schedule_next_tick(self, *, initial: bool = False) -> None:
        if self.ui is None:
            return
        self._cancel_tick()
        if initial:
            self.last_tick_started_at = None
        interval_ms = max(1, int(1000 / self.config.video.target_fps))
        self.tick_after_id = self.ui.schedule(interval_ms, self._tick)

    def _cancel_tick(self) -> None:
        if self.ui is not None and self.tick_after_id is not None:
            self.ui.cancel_scheduled(self.tick_after_id)
        self.tick_after_id = None

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def _sync_ui_state(self) -> None:
        if self.ui is None:
            return

        is_running = self.tick_after_id is not None
        can_select = self.current_frame is not None and not is_running
        can_start = (
            not is_running and self.selected_bbox is not None and self.state in {"Target Selected", "Paused", "Lost", "Uncertain"}
        )
        can_pause = is_running and self.state in {"Tracking", "Lost", "Uncertain"}
        can_reset = self.capture is not None

        self.ui.set_button_states(
            open_enabled=True,
            select_enabled=can_select,
            start_enabled=can_start,
            pause_enabled=can_pause,
            reset_enabled=can_reset,
        )
        self.ui.update_status(
            fps=self.last_fps,
            confidence=self.last_confidence,
            latency_ms=self.last_latency_ms,
            state=self.state,
        )
        self.ui.set_hint(self._hint_for_state())

    def _render_current_frame(self) -> None:
        if self.current_frame is None or self.ui is None:
            return
        frame = self.current_frame.copy()
        self._draw_overlay(frame)
        self.ui.show_frame(frame)

    def _draw_overlay(self, frame: np.ndarray) -> None:
        if self.current_bbox is not None:
            color = self._overlay_color()
            x, y, w, h = self.current_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if self.config.overlay.show_confidence and self.last_confidence is not None:
                label = f"{self.state} {self.last_confidence:.2f}"
                cv2.putText(frame, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.config.overlay.show_trail and len(self.trail) > 1:
            for index in range(1, len(self.trail)):
                cv2.line(frame, self.trail[index - 1], self.trail[index], (77, 163, 255), 2)

    def _overlay_color(self) -> tuple[int, int, int]:
        if self.state == "Lost":
            return (64, 64, 255)
        if self.state == "Uncertain":
            return (0, 215, 255)
        return (80, 220, 120)

    def _append_trail_point(self, bbox: BBox) -> None:
        x, y, w, h = bbox
        self.trail.append((x + w // 2, y + h // 2))

    def _hint_for_state(self) -> str:
        if self.state == "Idle":
            return "Open a video to begin"
        if self.state == "Video Loaded":
            return "Click the target center, then drag to resize the target box"
        if self.state == "Target Selected":
            return "Press Start to begin tracking, or Select Target to reselect"
        if self.state == "Tracking":
            return "Tracking is running. Press Pause to stop or Reset to restart"
        if self.state == "Paused":
            return "Press Start to continue, or Reset to return to the first frame"
        if self.state == "Uncertain":
            return "Tracker confidence is weaker. Pause, reselect, or continue watching"
        if self.state == "Lost":
            return "Target is lost. Pause and reselect, or Reset to start over"
        return "Use the controls on the right to continue"

    def _send_drone_command(self) -> None:
        if self.ui is None:
            return
        if self.state not in {"Tracking", "Uncertain"}:
            return
        try:
            velocity = getattr(self.backend, "velocity", None)
            if velocity is None:
                return
            raw_vx = float(velocity[0])
            raw_vy = float(velocity[1])
            if self._smoothed_velocity is None:
                self._smoothed_velocity = np.array([raw_vx, raw_vy], dtype=np.float32)
            else:
                self._smoothed_velocity = self._smoothed_velocity * 0.7 + np.array([raw_vx, raw_vy]) * 0.3
            vx, vy = float(self._smoothed_velocity[0]), float(self._smoothed_velocity[1])
            speed = (vx * vx + vy * vy) ** 0.5
            if speed < 1.5:
                direction, force = "Stationary", 0
            else:
                abs_vx, abs_vy = abs(vx), abs(vy)
                if abs_vx > abs_vy:
                    direction = "Right" if vx > 0 else "Left"
                    force = min(100, int(abs_vx * 10))
                else:
                    direction = "Down" if vy > 0 else "Up"
                    force = min(100, int(abs_vy * 10))
            if self._last_command != (direction, force):
                self._last_command = (direction, force)
                print(f"COMMAND: {direction} {force}%")
                self.ui.set_command(direction, force)
        except (AttributeError, TypeError, ValueError):
            pass
