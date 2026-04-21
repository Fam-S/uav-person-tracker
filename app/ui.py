from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from app.config import AppConfig


class MainWindow:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.root = tk.Tk()
        self.root.title(config.app.title)
        self.root.geometry(f"{config.app.width}x{config.app.height}")
        self.root.minsize(config.app.min_width, config.app.min_height)
        self.root.configure(bg="#111111")

        self.controller = None
        self.selection_enabled = False
        self.selection_center_frame: tuple[float, float] | None = None
        self.selection_bbox_frame: tuple[int, int, int, int] | None = None
        self.persistent_bbox_frame: tuple[int, int, int, int] | None = None
        self.selection_rect_id: int | None = None
        self.template_rect_id: int | None = None
        self.search_rect_id: int | None = None
        self.selection_cross_ids: list[int] = []
        self.selection_label_id: int | None = None
        self.template_label_id: int | None = None
        self.search_label_id: int | None = None
        self.rendered_image = None
        self.image_id: int | None = None
        self.current_frame_size = (1, 1)
        self.display_box = (0, 0, 1, 1)

        self._build_layout()
        self._build_bindings()

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)

        self.video_canvas = tk.Canvas(self.root, bg="#0d0d0d", highlightthickness=0, cursor="crosshair")
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=(12, 6))

        self.sidebar = tk.Frame(self.root, bg="#1a1a1a", padx=12, pady=12)
        self.sidebar.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=(12, 6))
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(7, weight=1)

        self.status_bar = tk.Frame(self.root, bg="#151515", padx=12, pady=8)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))
        for index in range(6):
            self.status_bar.grid_columnconfigure(index, weight=1)

        button_style = {
            "bg": "#2a2a2a",
            "fg": "#f0f0f0",
            "activebackground": "#3a3a3a",
            "activeforeground": "#ffffff",
            "bd": 0,
            "relief": "flat",
            "padx": 10,
            "pady": 10,
        }
        label_style = {"bg": "#1a1a1a", "fg": "#f0f0f0", "anchor": "w"}

        self.open_button = tk.Button(self.sidebar, text="Open Video", command=self._on_open_video, **button_style)
        self.select_button = tk.Button(self.sidebar, text="Select Target", command=self._on_select_target, **button_style)
        self.start_button = tk.Button(self.sidebar, text="Start", command=self._on_start, **button_style)
        self.pause_button = tk.Button(self.sidebar, text="Pause", command=self._on_pause, **button_style)
        self.reset_button = tk.Button(self.sidebar, text="Reset", command=self._on_reset, **button_style)

        self.open_button.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.select_button.grid(row=1, column=0, sticky="ew", pady=8)
        self.start_button.grid(row=2, column=0, sticky="ew", pady=8)
        self.pause_button.grid(row=3, column=0, sticky="ew", pady=8)
        self.reset_button.grid(row=4, column=0, sticky="ew", pady=8)

        tk.Label(self.sidebar, text="Events", font=("Segoe UI", 11, "bold"), **label_style).grid(
            row=5, column=0, sticky="ew", pady=(16, 8)
        )
        self.events_list = tk.Listbox(
            self.sidebar,
            bg="#101010",
            fg="#e6e6e6",
            bd=0,
            highlightthickness=0,
            activestyle="none",
        )
        self.events_list.grid(row=6, column=0, rowspan=2, sticky="nsew")

        status_style = {"bg": "#151515", "fg": "#f0f0f0", "anchor": "w"}
        self.fps_var = tk.StringVar(value="FPS: --")
        self.confidence_var = tk.StringVar(value="Confidence: --")
        self.latency_var = tk.StringVar(value="Latency: --")
        self.state_var = tk.StringVar(value="State: Idle")
        self.backend_var = tk.StringVar(value="Backend: --")
        self.hint_var = tk.StringVar(value="Hint: Open a video to begin")

        tk.Label(self.status_bar, textvariable=self.fps_var, **status_style).grid(row=0, column=0, sticky="ew")
        tk.Label(self.status_bar, textvariable=self.confidence_var, **status_style).grid(row=0, column=1, sticky="ew")
        tk.Label(self.status_bar, textvariable=self.latency_var, **status_style).grid(row=0, column=2, sticky="ew")
        tk.Label(self.status_bar, textvariable=self.state_var, **status_style).grid(row=0, column=3, sticky="ew")
        tk.Label(self.status_bar, textvariable=self.backend_var, **status_style).grid(row=0, column=4, sticky="ew")
        tk.Label(self.status_bar, textvariable=self.hint_var, **status_style).grid(row=1, column=0, columnspan=6, sticky="ew", pady=(6, 0))

        self.set_button_states(
            open_enabled=True,
            select_enabled=False,
            start_enabled=False,
            pause_enabled=False,
            reset_enabled=False,
        )

    def _build_bindings(self) -> None:
        self.video_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.video_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.video_canvas.bind("<Configure>", self._on_canvas_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def bind_controller(self, controller) -> None:
        self.controller = controller

    def ask_video_path(self) -> str | None:
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.mpeg *.mpg"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Open Video", filetypes=filetypes)
        return path or None

    def show_frame(self, frame_bgr) -> None:
        frame_h, frame_w = frame_bgr.shape[:2]
        self.current_frame_size = (frame_w, frame_h)

        canvas_w = max(self.video_canvas.winfo_width(), 2)
        canvas_h = max(self.video_canvas.winfo_height(), 2)
        render_w, render_h, offset_x, offset_y = self._compute_render_box(frame_w, frame_h, canvas_w, canvas_h)
        self.display_box = (offset_x, offset_y, render_w, render_h)

        resized = cv2.resize(frame_bgr, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.rendered_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

        if self.image_id is None:
            self.image_id = self.video_canvas.create_image(
                offset_x, offset_y, anchor="nw", image=self.rendered_image, tags="frame"
            )
        else:
            self.video_canvas.itemconfig(self.image_id, image=self.rendered_image)
            self.video_canvas.coords(self.image_id, offset_x, offset_y)
        self._redraw_selection_overlay()

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
        self.fps_var.set(f"FPS: {fps:.1f}" if fps is not None else "FPS: --")
        self.confidence_var.set(
            f"Confidence: {confidence:.2f}" if confidence is not None else "Confidence: --"
        )
        self.latency_var.set(f"Latency: {latency_ms:.1f} ms" if latency_ms is not None else "Latency: --")
        self.state_var.set(f"State: {state}")

    def add_event(self, message: str) -> None:
        self.events_list.insert(0, message)
        while self.events_list.size() > 50:
            self.events_list.delete(tk.END)

    def set_hint(self, message: str) -> None:
        self.hint_var.set(f"Hint: {message}")

    def set_backend_name(self, backend_name: str) -> None:
        self.backend_var.set(f"Backend: {backend_name}")

    def set_command(self, direction: str, force: float) -> None:
        msg = "Stay" if direction == "Stationary" else f"{direction} {force:.0f}%"
        self.events_list.insert(0, msg)
        while self.events_list.size() > 50:
            self.events_list.delete(tk.END)

    def enable_target_selection(self) -> None:
        self.selection_enabled = True
        self.selection_center_frame = None
        self.selection_bbox_frame = self.persistent_bbox_frame
        self._clear_selection_overlay()
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
        messagebox.showerror(title, message)

    def show_info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message)

    def schedule(self, delay_ms: int, callback):
        return self.root.after(delay_ms, callback)

    def cancel_scheduled(self, callback_id) -> None:
        if callback_id is not None:
            self.root.after_cancel(callback_id)

    def run(self) -> None:
        self.root.mainloop()

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

    def _on_close(self) -> None:
        if self.controller is not None:
            self.controller.shutdown()
        self.root.destroy()

    def _on_canvas_resize(self, event) -> None:
        _ = event
        if self.controller is not None:
            self.controller.refresh_current_frame()

    def _on_canvas_press(self, event) -> None:
        if not self.selection_enabled:
            return
        frame_point = self._display_point_to_frame_point(event.x, event.y)
        if frame_point is None:
            return
        self.selection_center_frame = frame_point
        self.selection_bbox_frame = self._default_selection_bbox(frame_point)
        self._redraw_selection_overlay()

    def _on_canvas_drag(self, event) -> None:
        if not self.selection_enabled or self.selection_center_frame is None:
            return
        frame_point = self._display_point_to_frame_point(event.x, event.y)
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

    def _display_point_to_frame_point(self, x: int, y: int) -> tuple[float, float] | None:
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

        half_w = max(abs(drag_point[0] - center_x), 6.0)
        half_h = half_w / aspect_ratio

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
        self, center: tuple[float, float], width: int, height: int
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

    def _ensure_overlay_items(self) -> None:
        if self.template_rect_id is not None:
            return
        self.template_rect_id = self.video_canvas.create_rectangle(
            0, 0, 1, 1, outline="#7be495", width=2, dash=(6, 4), state="hidden"
        )
        self.search_rect_id = self.video_canvas.create_rectangle(
            0, 0, 1, 1, outline="#5aa0ff", width=2, dash=(8, 6), state="hidden"
        )
        self.selection_rect_id = self.video_canvas.create_rectangle(
            0, 0, 1, 1, outline="#4fd36d", width=2, state="hidden"
        )
        self.selection_cross_ids = [
            self.video_canvas.create_line(0, 0, 1, 1, fill="#4fd36d", width=2, state="hidden"),
            self.video_canvas.create_line(0, 0, 1, 1, fill="#4fd36d", width=2, state="hidden"),
        ]
        self.selection_label_id = self.video_canvas.create_text(
            0, 0, anchor="sw", text="Target", fill="#4fd36d",
            font=("Segoe UI", 10, "bold"), state="hidden",
        )
        self.template_label_id = self.video_canvas.create_text(
            0, 0, anchor="sw", text="Template Crop", fill="#7be495",
            font=("Segoe UI", 10), state="hidden",
        )
        self.search_label_id = self.video_canvas.create_text(
            0, 0, anchor="sw", text="Search Crop", fill="#5aa0ff",
            font=("Segoe UI", 10), state="hidden",
        )

    def _redraw_selection_overlay(self) -> None:
        overlay_bbox = self.selection_bbox_frame or self.persistent_bbox_frame
        if overlay_bbox is None:
            self._clear_selection_overlay()
            return

        target_box = self._frame_bbox_to_display_bbox(overlay_bbox)
        template_box = self._frame_bbox_to_display_bbox(self._template_crop_from_target(overlay_bbox))
        search_box = self._frame_bbox_to_display_bbox(self._search_crop_from_target(overlay_bbox))
        if target_box is None or template_box is None or search_box is None:
            self._clear_selection_overlay()
            return

        self._ensure_overlay_items()
        cx, cy, cw, ch = template_box
        sx, sy, sw, sh = search_box
        tx, ty, tw, th = target_box

        self.video_canvas.coords(self.template_rect_id, cx, cy, cx + cw, cy + ch)
        self.video_canvas.coords(self.search_rect_id, sx, sy, sx + sw, sy + sh)
        self.video_canvas.coords(self.selection_rect_id, tx, ty, tx + tw, ty + th)
        self.video_canvas.coords(self.selection_cross_ids[0], tx, ty, tx + tw, ty + th)
        self.video_canvas.coords(self.selection_cross_ids[1], tx + tw, ty, tx, ty + th)
        self.video_canvas.coords(self.selection_label_id, tx, max(16, ty - 10))
        self.video_canvas.coords(self.template_label_id, cx, max(16, cy - 10))
        self.video_canvas.coords(self.search_label_id, sx, max(16, sy - 10))

        for item_id in [
            self.template_rect_id, self.search_rect_id, self.selection_rect_id,
            self.selection_label_id, self.template_label_id, self.search_label_id,
            *self.selection_cross_ids,
        ]:
            self.video_canvas.itemconfig(item_id, state="normal")

    def _clear_selection_overlay(self) -> None:
        for item_id in [
            self.template_rect_id, self.search_rect_id, self.selection_rect_id,
            self.selection_label_id, self.template_label_id, self.search_label_id,
        ]:
            if item_id is not None:
                self.video_canvas.itemconfig(item_id, state="hidden")
        for item_id in self.selection_cross_ids:
            self.video_canvas.itemconfig(item_id, state="hidden")

    def _template_crop_from_target(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        side = int(round(max(w, h) * self.config.tracking.template_crop_scale))
        return self._square_box_from_center((x + w / 2, y + h / 2), max(side, max(w, h)))

    def _search_crop_from_target(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        tx, ty, tw, th = self._template_crop_from_target(bbox)
        side = int(round(max(tw, th) * self.config.tracking.search_crop_scale))
        return self._square_box_from_center((tx + tw / 2, ty + th / 2), max(side, max(tw, th)))

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

    def _frame_bbox_to_display_bbox(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        offset_x, offset_y, render_w, render_h = self.display_box
        frame_w, frame_h = self.current_frame_size
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

    def _compute_render_box(self, frame_w: int, frame_h: int, canvas_w: int, canvas_h: int) -> tuple[int, int, int, int]:
        if not self.config.video.keep_aspect_ratio:
            return (canvas_w, canvas_h, 0, 0)

        frame_ratio = frame_w / frame_h
        canvas_ratio = canvas_w / canvas_h
        if frame_ratio > canvas_ratio:
            render_w = canvas_w
            render_h = max(1, int(canvas_w / frame_ratio))
        else:
            render_h = canvas_h
            render_w = max(1, int(canvas_h * frame_ratio))

        offset_x = max(0, (canvas_w - render_w) // 2)
        offset_y = max(0, (canvas_h - render_h) // 2)
        return (render_w, render_h, offset_x, offset_y)

    @staticmethod
    def _set_button_state(button: tk.Button, enabled: bool) -> None:
        button.configure(state=(tk.NORMAL if enabled else tk.DISABLED))

    def set_window_title(self, title: str) -> None:
        self.root.title(title)

    def set_video_label(self, video_path: str | None) -> None:
        if video_path is None:
            self.set_window_title(self.config.app.title)
            return
        self.set_window_title(f"{self.config.app.title} - {Path(video_path).name}")
