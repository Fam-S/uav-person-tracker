from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def _state_color(state: str) -> Tuple[int, int, int]:
    """
    Return BGR color by tracking state.
    """
    state = state.lower()

    if state == "tracking":
        return (0, 255, 0)      # green
    if state == "uncertain":
        return (0, 255, 255)    # yellow
    if state == "lost":
        return (0, 0, 255)      # red

    return (255, 255, 255)      # white fallback


def _clip_point(x: float, y: float, frame_w: int, frame_h: int) -> Tuple[int, int]:
    x = int(round(max(0, min(x, frame_w - 1))))
    y = int(round(max(0, min(y, frame_h - 1))))
    return x, y


def draw_tracking_result(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    score: float,
    state: str,
    search_bbox: Tuple[int, int, int, int] | None = None,
    grid_pos: Tuple[int, int] | None = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw tracking result on a copy of frame.

    Args:
        frame: input image [H, W, 3]
        bbox: target bbox in format (x, y, w, h)
        score: confidence score
        state: Tracking / Uncertain / Lost
        search_bbox: optional search region bbox
        grid_pos: optional best grid position
    """
    if frame is None:
        raise ValueError("Frame is None.")

    output = frame.copy()
    frame_h, frame_w = output.shape[:2]

    color = _state_color(state)

    # ---- draw target bbox ----
    x, y, w, h = bbox
    x1, y1 = _clip_point(x, y, frame_w, frame_h)
    x2, y2 = _clip_point(x + w, y + h, frame_w, frame_h)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

    # ---- label text ----
    label = f"{state} | score={score:.3f}"
    if grid_pos is not None:
        label += f" | grid={grid_pos}"

    text_x = x1
    text_y = max(20, y1 - 10)

    cv2.putText(
        output,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )

    # ---- draw search region (optional) ----
    if search_bbox is not None:
        sx, sy, sw, sh = search_bbox
        sx1, sy1 = _clip_point(sx, sy, frame_w, frame_h)
        sx2, sy2 = _clip_point(sx + sw, sy + sh, frame_w, frame_h)

        cv2.rectangle(output, (sx1, sy1), (sx2, sy2), (255, 0, 0), 1)

        cv2.putText(
            output,
            "search",
            (sx1, max(20, sy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_trail(
    frame: np.ndarray,
    centers: Iterable[Tuple[float, float]],
    color: Tuple[int, int, int] = (255, 0, 255),
    radius: int = 2,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw target trajectory trail on a copy of frame.
    """
    if frame is None:
        raise ValueError("Frame is None.")

    output = frame.copy()
    points = list(centers)

    if len(points) == 0:
        return output

    frame_h, frame_w = output.shape[:2]

    clipped_points = [
        _clip_point(x, y, frame_w, frame_h)
        for x, y in points
    ]

    for i in range(1, len(clipped_points)):
        cv2.line(output, clipped_points[i - 1], clipped_points[i], color, thickness)

    for pt in clipped_points:
        cv2.circle(output, pt, radius, color, -1)

    return output


def draw_tracking_overlay(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    score: float,
    state: str,
    search_bbox: Tuple[int, int, int, int] | None = None,
    grid_pos: Tuple[int, int] | None = None,
    trail: Iterable[Tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Full visualization helper:
    - bbox
    - score
    - state
    - optional search box
    - optional trail
    """
    output = draw_tracking_result(
        frame=frame,
        bbox=bbox,
        score=score,
        state=state,
        search_bbox=search_bbox,
        grid_pos=grid_pos,
    )

    if trail is not None:
        output = draw_trail(output, trail)

    return output


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Return bbox center from (x, y, w, h).
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)