from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from data.crop_utils import compute_crop_size


SIAMAPN_CONTEXT_AMOUNT = 0.5
SIAMAPN_TEMPLATE_SIZE = 127
SIAMAPN_SEARCH_SIZE = 287


class VideoWidget(QWidget):
    """Custom video surface that renders the frame and selection overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._frame_bgr: np.ndarray | None = None
        self._frame_buffer: np.ndarray | None = None
        self._qimage: QImage | None = None
        self._frame_size = (1, 1)
        self._display_box = (0, 0, 1, 1)
        self._target_bbox: tuple[int, int, int, int] | None = None
        self._template_bbox: tuple[int, int, int, int] | None = None
        self._search_bbox: tuple[int, int, int, int] | None = None
        self._keep_aspect_ratio = True
        self.on_mouse_press = None
        self.on_mouse_move = None
        self.on_mouse_release = None

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    @property
    def display_box(self) -> tuple[int, int, int, int]:
        return self._display_box

    def set_keep_aspect_ratio(self, enabled: bool) -> None:
        self._keep_aspect_ratio = enabled
        self._render_frame()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self._frame_bgr = frame_bgr
        self._render_frame()

    def refresh(
        self,
        frame_bgr: np.ndarray,
        target_bbox: tuple[int, int, int, int] | None,
        template_bbox: tuple[int, int, int, int] | None,
        search_bbox: tuple[int, int, int, int] | None,
    ) -> None:
        """Set frame + overlay in one shot — single repaint."""
        self._frame_bgr = frame_bgr
        frame_h, frame_w = frame_bgr.shape[:2]
        self._frame_size = (frame_w, frame_h)
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        render_w, render_h, offset_x, offset_y = self._compute_render_box(frame_w, frame_h, widget_w, widget_h)
        self._display_box = (offset_x, offset_y, render_w, render_h)
        resized = cv2.resize(frame_bgr, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        self._frame_buffer = resized
        self._qimage = QImage(resized.data, render_w, render_h, resized.strides[0], QImage.Format.Format_BGR888)
        self._target_bbox = target_bbox
        self._template_bbox = template_bbox
        self._search_bbox = search_bbox
        self.update()

    def set_selection_overlay(
        self,
        target_bbox: tuple[int, int, int, int] | None,
        template_bbox: tuple[int, int, int, int] | None,
        search_bbox: tuple[int, int, int, int] | None,
    ) -> None:
        self._target_bbox = target_bbox
        self._template_bbox = template_bbox
        self._search_bbox = search_bbox
        self.update()

    def clear_selection_overlay(self) -> None:
        self._target_bbox = None
        self._template_bbox = None
        self._search_bbox = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d0d0d"))

        if self._qimage is not None:
            offset_x, offset_y, _, _ = self._display_box
            painter.drawImage(QPoint(offset_x, offset_y), self._qimage)
            self._paint_overlays(painter)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._render_frame()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.on_mouse_press is not None:
            self.on_mouse_press(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.on_mouse_move is not None:
            self.on_mouse_move(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.on_mouse_release is not None:
            self.on_mouse_release(event)

    def _render_frame(self) -> None:
        if self._frame_bgr is None:
            self._frame_size = (1, 1)
            self._display_box = (0, 0, 1, 1)
            self._qimage = None
            self._frame_buffer = None
            self.update()
            return

        frame_h, frame_w = self._frame_bgr.shape[:2]
        self._frame_size = (frame_w, frame_h)

        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        render_w, render_h, offset_x, offset_y = self._compute_render_box(frame_w, frame_h, widget_w, widget_h)
        self._display_box = (offset_x, offset_y, render_w, render_h)

        resized = cv2.resize(self._frame_bgr, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        self._frame_buffer = resized
        self._qimage = QImage(
            resized.data,
            render_w,
            render_h,
            resized.strides[0],
            QImage.Format.Format_BGR888,
        )
        self.update()

    def _paint_overlays(self, painter: QPainter) -> None:
        if self._template_bbox is not None:
            self._draw_rect(painter, self._template_bbox, "#7be495", dash=True, label="Template Crop")
        if self._search_bbox is not None:
            self._draw_rect(painter, self._search_bbox, "#5aa0ff", dash=True, label="Search Crop")
        if self._target_bbox is not None:
            self._draw_rect(painter, self._target_bbox, "#4fd36d", label="Target")
            target_box = self._frame_bbox_to_display_bbox(self._target_bbox)
            if target_box is not None:
                dx, dy, dw, dh = target_box
                pen = QPen(QColor("#4fd36d"), 2)
                painter.setPen(pen)
                painter.drawLine(dx, dy, dx + dw, dy + dh)
                painter.drawLine(dx + dw, dy, dx, dy + dh)

    def _draw_rect(
        self,
        painter: QPainter,
        bbox: tuple[int, int, int, int],
        color_hex: str,
        *,
        dash: bool = False,
        label: str = "",
    ) -> None:
        display_bbox = self._frame_bbox_to_display_bbox(bbox)
        if display_bbox is None:
            return

        dx, dy, dw, dh = display_bbox
        pen = QPen(QColor(color_hex), 2)
        if dash:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRect(dx, dy, dw, dh))
        if label:
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(QPoint(dx, max(16, dy - 10)), label)

    def _frame_bbox_to_display_bbox(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        offset_x, offset_y, render_w, render_h = self._display_box
        frame_w, frame_h = self._frame_size
        if render_w <= 1 or render_h <= 1 or frame_w <= 0 or frame_h <= 0:
            return None

        x, y, w, h = bbox
        scale_x = render_w / frame_w
        scale_y = render_h / frame_h
        dx = int(round(offset_x + x * scale_x))
        dy = int(round(offset_y + y * scale_y))
        dw = max(1, int(round(w * scale_x)))
        dh = max(1, int(round(h * scale_y)))
        return (dx, dy, dw, dh)

    def _compute_render_box(self, frame_w: int, frame_h: int, widget_w: int, widget_h: int) -> tuple[int, int, int, int]:
        if not self._keep_aspect_ratio:
            return (widget_w, widget_h, 0, 0)

        frame_ratio = frame_w / frame_h
        widget_ratio = widget_w / widget_h
        if frame_ratio > widget_ratio:
            render_w = widget_w
            render_h = max(1, int(widget_w / frame_ratio))
        else:
            render_h = widget_h
            render_w = max(1, int(widget_h * frame_ratio))

        offset_x = max(0, (widget_w - render_w) // 2)
        offset_y = max(0, (widget_h - render_h) // 2)
        return (render_w, render_h, offset_x, offset_y)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.controller = None
        self.selection_enabled = False
        self.selection_center_frame: tuple[float, float] | None = None
        self.selection_bbox_frame: tuple[int, int, int, int] | None = None
        self.persistent_bbox_frame: tuple[int, int, int, int] | None = None
        self.current_frame_size = (1, 1)
        self.display_box = (0, 0, 1, 1)

        self.setWindowTitle(config.app.title)
        self.resize(config.app.width, config.app.height)
        self.setMinimumSize(config.app.min_width, config.app.min_height)

        self._tick_timer = QTimer(self)
        self._tick_timer.setSingleShot(True)
        self._tick_callback = None
        self._tick_timer.timeout.connect(self._on_tick_timer)

        self._build_layout()
        self._build_bindings()
        self.set_button_states(
            open_enabled=True,
            select_enabled=False,
            start_enabled=False,
            pause_enabled=False,
            reset_enabled=False,
        )

    def _build_layout(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        self.video_widget = VideoWidget()
        self.video_widget.setObjectName("video_widget")
        self.video_widget.set_keep_aspect_ratio(self.config.video.keep_aspect_ratio)
        root_layout.addWidget(self.video_widget, stretch=3)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(8)

        button_style = """
            QPushButton {
                background-color: #2a2a2a;
                color: #f0f0f0;
                border: 0;
                padding: 10px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #1f1f1f;
                color: #808080;
            }
        """
        sidebar.setStyleSheet(
            """
            QWidget#sidebar {
                background-color: #1a1a1a;
                color: #f0f0f0;
            }
            QListWidget {
                background-color: #101010;
                color: #e6e6e6;
                border: 0;
            }
            """
        )

        self.open_button = QPushButton("Open Video")
        self.select_button = QPushButton("Select Target")
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        for button in [self.open_button, self.select_button, self.start_button, self.pause_button, self.reset_button]:
            button.setStyleSheet(button_style)
            sidebar_layout.addWidget(button)

        events_label = QLabel("Events")
        events_label.setStyleSheet("color: #f0f0f0; font-weight: bold;")
        sidebar_layout.addWidget(events_label)

        self.events_list = QListWidget()
        sidebar_layout.addWidget(self.events_list, stretch=1)
        root_layout.addWidget(sidebar, stretch=1)

        status_bar = QStatusBar(self)
        status_bar.setStyleSheet(
            """
            QStatusBar {
                background-color: #151515;
                color: #f0f0f0;
            }
            """
        )
        self.setStatusBar(status_bar)

        status_style = "color: #f0f0f0;"
        self.fps_label = QLabel("FPS: --")
        self.confidence_label = QLabel("Confidence: --")
        self.latency_label = QLabel("Latency: --")
        self.state_label = QLabel("State: Idle")
        self.backend_label = QLabel("Backend: --")
        self.hint_label = QLabel("Hint: Open a video to begin")
        for label in [
            self.fps_label,
            self.confidence_label,
            self.latency_label,
            self.state_label,
            self.backend_label,
        ]:
            label.setStyleSheet(status_style)
            status_bar.addWidget(label)
        self.hint_label.setStyleSheet(status_style)
        status_bar.addPermanentWidget(self.hint_label, 1)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #111111;
            }
            QWidget#central {
                background-color: #111111;
            }
            """
        )

        self.open_button.clicked.connect(self._on_open_video)
        self.select_button.clicked.connect(self._on_select_target)
        self.start_button.clicked.connect(self._on_start)
        self.pause_button.clicked.connect(self._on_pause)
        self.reset_button.clicked.connect(self._on_reset)

    def _build_bindings(self) -> None:
        self.video_widget.on_mouse_press = self._on_canvas_press
        self.video_widget.on_mouse_move = self._on_canvas_drag
        self.video_widget.on_mouse_release = self._on_canvas_release

    def _on_open_video(self) -> None:
        if self.controller is not None:
            self.controller.open_video()

    def _on_select_target(self) -> None:
        if self.controller is not None:
            self.controller.enable_target_selection()

    def _on_start(self) -> None:
        if self.controller is not None:
            self.controller.start_tracking()

    def _on_pause(self) -> None:
        if self.controller is not None:
            self.controller.pause_tracking()

    def _on_reset(self) -> None:
        if self.controller is not None:
            self.controller.reset_tracking()

    def bind_controller(self, controller) -> None:
        self.controller = controller

    def ask_video_path(self) -> str | None:
        filetypes = "Video files (*.mp4 *.avi *.mov *.mkv *.mpeg *.mpg);;All files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", filetypes)
        return path or None

    def show_frame(self, frame_bgr) -> None:
        overlay = self.selection_bbox_frame or self.persistent_bbox_frame
        if overlay is not None:
            template = self._template_crop_from_target(overlay)
            search = self._search_crop_from_target(overlay)
        else:
            template = search = None
        self.video_widget.refresh(frame_bgr, overlay, template, search)
        self.current_frame_size = self.video_widget.frame_size
        self.display_box = self.video_widget.display_box

    def set_button_states(
        self,
        *,
        open_enabled: bool,
        select_enabled: bool,
        start_enabled: bool,
        pause_enabled: bool,
        reset_enabled: bool,
    ) -> None:
        self._set_button_state(self.open_button, open_enabled)
        self._set_button_state(self.select_button, select_enabled)
        self._set_button_state(self.start_button, start_enabled)
        self._set_button_state(self.pause_button, pause_enabled)
        self._set_button_state(self.reset_button, reset_enabled)

    def update_status(self, *, fps: float | None, confidence: float | None, latency_ms: float | None, state: str) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}" if fps is not None else "FPS: --")
        self.confidence_label.setText(
            f"Confidence: {confidence:.2f}" if confidence is not None else "Confidence: --"
        )
        self.latency_label.setText(f"Latency: {latency_ms:.1f} ms" if latency_ms is not None else "Latency: --")
        self.state_label.setText(f"State: {state}")

    def add_event(self, message: str) -> None:
        self.events_list.insertItem(0, message)
        while self.events_list.count() > 50:
            self.events_list.takeItem(self.events_list.count() - 1)

    def set_hint(self, message: str) -> None:
        self.hint_label.setText(f"Hint: {message}")

    def set_backend_name(self, backend_name: str) -> None:
        self.backend_label.setText(f"Backend: {backend_name}")

    def set_command(self, direction: str, force: float) -> None:
        msg = "Stay" if direction == "Stationary" else f"{direction} {force:.0f}%"
        self.add_event(msg)

    def enable_target_selection(self) -> None:
        self.selection_enabled = True
        self.selection_center_frame = None
        self.selection_bbox_frame = self.persistent_bbox_frame
        self._redraw_selection_overlay()
        self.add_event("Click target center, then drag to resize")
        self.set_hint("Click the target center, then drag to resize the target box")

    def clear_selection_box(self) -> None:
        self.selection_enabled = False
        self.selection_center_frame = None
        self.selection_bbox_frame = None
        self.persistent_bbox_frame = None
        self._clear_selection_overlay()

    def set_persistent_selection_box(self, bbox: tuple[int, int, int, int] | None) -> None:
        self.persistent_bbox_frame = bbox
        if not self.selection_enabled:
            self.selection_bbox_frame = bbox
        self._redraw_selection_overlay()

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _on_tick_timer(self) -> None:
        if self._tick_callback is not None:
            self._tick_callback()

    def schedule(self, delay_ms: int, callback) -> QTimer:
        self._tick_callback = callback
        self._tick_timer.start(delay_ms)
        return self._tick_timer

    def cancel_scheduled(self, _timer: QTimer | None) -> None:
        self._tick_timer.stop()
        self._tick_callback = None

    def run(self) -> None:
        pass

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.controller is not None:
            self.controller.shutdown()
        event.accept()

    def set_window_title(self, title: str) -> None:
        self.setWindowTitle(title)

    def set_video_label(self, video_path: str | None) -> None:
        if video_path is None:
            self.set_window_title(self.config.app.title)
            return
        self.set_window_title(f"{self.config.app.title} - {Path(video_path).name}")

    def _on_canvas_press(self, event) -> None:
        if not self.selection_enabled:
            return
        frame_point = self._display_point_to_frame_point(event.position().x(), event.position().y())
        if frame_point is None:
            return
        self.selection_center_frame = frame_point
        self.selection_bbox_frame = self._default_selection_bbox(frame_point)
        self._redraw_selection_overlay()

    def _on_canvas_drag(self, event) -> None:
        if not self.selection_enabled or self.selection_center_frame is None:
            return
        frame_point = self._display_point_to_frame_point(event.position().x(), event.position().y())
        if frame_point is None:
            return
        self.selection_bbox_frame = self._selection_bbox_from_drag(frame_point)
        self._redraw_selection_overlay()

    def _on_canvas_release(self, event) -> None:
        if not self.selection_enabled or self.selection_bbox_frame is None:
            return
        _ = event
        frame_bbox = self._sanitize_frame_bbox(self.selection_bbox_frame)
        if frame_bbox is None:
            self.clear_selection_box()
            self.add_event("Selection ignored: box too small or outside frame")
            return
        self.selection_enabled = False
        self.persistent_bbox_frame = frame_bbox
        self.selection_bbox_frame = frame_bbox
        self._redraw_selection_overlay()
        if self.controller is not None:
            self.controller.on_target_selected(frame_bbox)

    def _display_point_to_frame_point(self, x: float, y: float) -> tuple[float, float] | None:
        offset_x, offset_y, render_w, render_h = self.display_box
        frame_w, frame_h = self.current_frame_size
        if render_w <= 1 or render_h <= 1:
            return None
        if not (offset_x <= x <= offset_x + render_w and offset_y <= y <= offset_y + render_h):
            return None
        scale_x = frame_w / render_w
        scale_y = frame_h / render_h
        frame_x = (x - offset_x) * scale_x
        frame_y = (y - offset_y) * scale_y
        return (frame_x, frame_y)

    def _default_selection_bbox(self, center: tuple[float, float]) -> tuple[int, int, int, int]:
        frame_w, frame_h = self.current_frame_size
        aspect_ratio = self.config.tracking.selection_aspect_ratio
        default_h = max(24, int(frame_h * self.config.tracking.default_selection_height_fraction))
        default_w = max(12, int(default_h * aspect_ratio))
        bbox = self._bbox_from_center_and_size(center, default_w, default_h)
        return self._fit_bbox_to_frame(bbox)

    def _selection_bbox_from_drag(self, drag_point: tuple[float, float]) -> tuple[int, int, int, int]:
        frame_w, frame_h = self.current_frame_size
        center_x, center_y = self.selection_center_frame
        aspect_ratio = self.config.tracking.selection_aspect_ratio

        dx = abs(drag_point[0] - center_x)
        dy = abs(drag_point[1] - center_y)
        half_h = max(dy, dx / aspect_ratio, 12.0)
        half_w = max(half_h * aspect_ratio, 6.0)

        max_half_w = min(center_x, frame_w - center_x)
        max_half_h = min(center_y, frame_h - center_y)

        if half_h > max_half_h:
            half_h = max(6.0, max_half_h)
            half_w = half_h * aspect_ratio
        if half_w > max_half_w:
            half_w = max(6.0, max_half_w)
            half_h = half_w / aspect_ratio

        width = max(12, int(half_w * 2.0))
        height = max(24, int(half_h * 2.0))
        return self._fit_bbox_to_frame(self._bbox_from_center_and_size(self.selection_center_frame, width, height))

    def _bbox_from_center_and_size(
        self,
        center: tuple[float, float],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        center_x, center_y = center
        x = int(round(center_x - width / 2))
        y = int(round(center_y - height / 2))
        return (x, y, int(width), int(height))

    def _fit_bbox_to_frame(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        frame_w, frame_h = self.current_frame_size
        center_x = x + w / 2
        center_y = y + h / 2
        aspect_ratio = self.config.tracking.selection_aspect_ratio

        max_half_w = max(6.0, min(center_x, frame_w - center_x))
        max_half_h = max(12.0, min(center_y, frame_h - center_y))

        half_w = min(w / 2, max_half_w, max_half_h * aspect_ratio)
        half_h = half_w / aspect_ratio

        width = max(12, int(half_w * 2.0))
        height = max(24, int(half_h * 2.0))
        x = int(round(center_x - width / 2))
        y = int(round(center_y - height / 2))
        x = max(0, min(x, frame_w - width))
        y = max(0, min(y, frame_h - height))
        return (x, y, width, height)

    def _sanitize_frame_bbox(self, bbox: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        if bbox is None:
            return None
        x, y, w, h = self._fit_bbox_to_frame(bbox)
        if w < 12 or h < 24:
            return None
        return (x, y, w, h)

    def _redraw_selection_overlay(self) -> None:
        overlay_bbox = self.selection_bbox_frame or self.persistent_bbox_frame
        if overlay_bbox is None:
            self._clear_selection_overlay()
            return

        template_box = self._template_crop_from_target(overlay_bbox)
        search_box = self._search_crop_from_target(overlay_bbox)
        self.video_widget.set_selection_overlay(overlay_bbox, template_box, search_box)

    def _clear_selection_overlay(self) -> None:
        self.video_widget.clear_selection_overlay()

    def _template_crop_from_target(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        side = int(
            round(
                compute_crop_size(
                    bbox,
                    context_amount=SIAMAPN_CONTEXT_AMOUNT,
                    area_scale=self.config.tracking.template_crop_scale,
                )
            )
        )
        return self._square_box_from_center((x + w / 2, y + h / 2), side)

    def _search_crop_from_target(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        if self.config.tracking.backend == "siamapn":
            area_scale = SIAMAPN_SEARCH_SIZE / SIAMAPN_TEMPLATE_SIZE
        else:
            area_scale = self.config.tracking.search_crop_scale
        side = int(
            round(
                compute_crop_size(
                    bbox,
                    context_amount=SIAMAPN_CONTEXT_AMOUNT,
                    area_scale=area_scale,
                )
            )
        )
        return self._square_box_from_center((x + w / 2, y + h / 2), side)

    def _square_box_from_center(self, center: tuple[float, float], side: int) -> tuple[int, int, int, int]:
        frame_w, frame_h = self.current_frame_size
        center_x, center_y = center
        half_side = max(6.0, side / 2)
        max_half_side = max(6.0, min(center_x, frame_w - center_x, center_y, frame_h - center_y))
        half_side = min(half_side, max_half_side)
        final_side = max(12, int(round(half_side * 2.0)))
        x = int(round(center_x - final_side / 2))
        y = int(round(center_y - final_side / 2))
        x = max(0, min(x, frame_w - final_side))
        y = max(0, min(y, frame_h - final_side))
        return (x, y, final_side, final_side)

    @staticmethod
    def _set_button_state(button: QPushButton, enabled: bool) -> None:
        button.setEnabled(enabled)
