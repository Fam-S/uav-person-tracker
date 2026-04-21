from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from app.config import TrackingSettings


BBox = tuple[int, int, int, int]

DEFAULT_MOTION_PREDICTION_BLEND = 0.25
DEFAULT_MAX_RELATIVE_CENTER_JUMP = 1.6
DEFAULT_LOST_STREAK = 5
DEFAULT_UNCERTAIN_STREAK = 2
DEFAULT_SEARCH_BASE_SCALE = 0.9
DEFAULT_SEARCH_MIN_MARGIN = 16
DEFAULT_SEARCH_MOTION_GAIN = 2.0
DEFAULT_OCCLUSION_SEARCH_GAIN = 2.5
DEFAULT_MOTION_ONLY_FRAMES = 3
DEFAULT_CONFIDENCE_EMA_ALPHA = 0.35
DEFAULT_CANDIDATE_MATCH_THRESHOLD = 0.50
DEFAULT_STRICT_JUMP_THRESHOLD = 1.50
DEFAULT_ALLOWED_RECOVERY_JUMP = 2.0
DEFAULT_VELOCITY_DECAY_ON_REJECT = 0.5
DEFAULT_RESPONSE_MARGIN_THRESHOLD = 0.02
DEFAULT_STRONG_RESPONSE_MARGIN = 0.06
DEFAULT_STRONG_MATCH_THRESHOLD = 0.62

DEFAULT_TEMPLATE_UPDATE_ALPHA = 0.03
DEFAULT_TEMPLATE_UPDATE_MIN_CONFIDENCE = 0.8
DEFAULT_TEMPLATE_UPDATE_MIN_RESPONSE_MARGIN = 0.10
DEFAULT_TEMPLATE_UPDATE_MAX_JUMP = 0.35
DEFAULT_TEMPLATE_UPDATE_INTERVAL = 8


@dataclass(slots=True)
class TrackingResult:
    bbox: BBox | None
    confidence: float
    state: str
    latency_ms: float


@dataclass(slots=True)
class Candidate:
    bbox: BBox
    score: float
    jump_factor: float
    match_score: float
    response_margin: float


class TraditionalTracker:
    """Lean traditional tracker for the app demo path.

    Optimized for speed:
    - single-scale local search
    - grayscale template matching only
    - light constant-velocity motion prior
    - frozen first-frame template
    - simple Tracking / Uncertain / Lost state logic
    """

    def __init__(self, settings: TrackingSettings) -> None:
        self.settings = settings
        self.initialized = False
        self.template_gray: np.ndarray | None = None
        self.bbox: BBox | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.prev_confidence: float | None = None
        self.confidence_ema: float | None = None
        self.low_confidence_streak = 0
        self.uncertain_streak = 0
        self._track_frame_count = 0
        self._occluded_streak = 0
        self._motion_only_frames = 0

    def load(self, checkpoint: str | None) -> None:
        _ = checkpoint

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        x, y, w, h = _clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        if w < 8 or h < 12:
            raise ValueError("Selected target is too small to track")

        self.bbox = (x, y, w, h)
        self._set_template_from_frame(frame, self.bbox)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.prev_confidence = 1.0
        self.confidence_ema = 1.0
        self.low_confidence_streak = 0
        self.uncertain_streak = 0
        self._track_frame_count = 0
        self._occluded_streak = 0
        self._motion_only_frames = 0
        self.initialized = True

    def track(self, frame: np.ndarray) -> TrackingResult:
        if not self.initialized or self.bbox is None or self.template_gray is None:
            return TrackingResult(None, 0.0, "Lost", 0.0)

        start = perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape[:2]

        predicted_bbox = self._predict_bbox(frame_w, frame_h)
        candidate = self._evaluate_candidate(gray, predicted_bbox)

        if candidate is None:
            self.low_confidence_streak += 1
            self.uncertain_streak += 1
            state = "Lost" if self.low_confidence_streak >= DEFAULT_LOST_STREAK else "Uncertain"
            latency_ms = (perf_counter() - start) * 1000.0
            return TrackingResult(self.bbox, 0.0, state, latency_ms)

        if not self._is_candidate_acceptable(candidate):
            self._check_occlusion(candidate)
            if self._occluded_streak > 0:
                return self._handle_occlusion(frame, gray, start, frame_h, frame_w)
            self.low_confidence_streak += 1
            self.uncertain_streak += 1
            self.velocity *= DEFAULT_VELOCITY_DECAY_ON_REJECT
            confidence = float(np.clip(candidate.score, 0.0, 1.0))
            if self.confidence_ema is None:
                self.confidence_ema = confidence
            else:
                self.confidence_ema = (
                    (1.0 - DEFAULT_CONFIDENCE_EMA_ALPHA) * self.confidence_ema
                    + DEFAULT_CONFIDENCE_EMA_ALPHA * confidence
                )
            state = "Lost" if self.low_confidence_streak >= DEFAULT_LOST_STREAK else "Uncertain"
            latency_ms = (perf_counter() - start) * 1000.0
            return TrackingResult(self.bbox, confidence, state, latency_ms)

        previous_bbox = self.bbox
        self.bbox = candidate.bbox
        self._update_velocity(previous_bbox, candidate.bbox)

        confidence = float(np.clip(candidate.score, 0.0, 1.0))
        if self.confidence_ema is None:
            self.confidence_ema = confidence
        else:
            self.confidence_ema = (
                (1.0 - DEFAULT_CONFIDENCE_EMA_ALPHA) * self.confidence_ema
                + DEFAULT_CONFIDENCE_EMA_ALPHA * confidence
            )

        state = self._state_from_observation(confidence, self.confidence_ema, candidate.jump_factor)

        if confidence < self.settings.uncertain_confidence_threshold:
            self.low_confidence_streak += 1
            self.uncertain_streak += 1
        else:
            self.low_confidence_streak = 0
            self.uncertain_streak = 0

        if self.low_confidence_streak >= DEFAULT_LOST_STREAK:
            state = "Lost"
        elif state == "Uncertain" and self.uncertain_streak < DEFAULT_UNCERTAIN_STREAK:
            state = "Tracking"

        if state == "Tracking":
            self._maybe_update_template(gray, candidate)

        self.prev_confidence = confidence
        latency_ms = (perf_counter() - start) * 1000.0
        return TrackingResult(self.bbox, confidence, state, latency_ms)

    def reset(self) -> None:
        self.initialized = False
        self.template_gray = None
        self.bbox = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.prev_confidence = None
        self.confidence_ema = None
        self.low_confidence_streak = 0
        self.uncertain_streak = 0
        self._track_frame_count = 0
        self._occluded_streak = 0
        self._motion_only_frames = 0

    def _predict_bbox(self, frame_w: int, frame_h: int) -> BBox:
        assert self.bbox is not None
        x, y, w, h = self.bbox
        px = int(round(x + self.velocity[0]))
        py = int(round(y + self.velocity[1]))
        return _clamp_bbox((px, py, w, h), frame_w, frame_h)

    def _evaluate_candidate(self, gray: np.ndarray, predicted_bbox: BBox) -> Candidate | None:
        assert self.template_gray is not None and self.bbox is not None
        px, py, pw, ph = predicted_bbox
        frame_h, frame_w = gray.shape[:2]

        base_margin_x = int(round(pw * self.settings.search_radius_scale * DEFAULT_SEARCH_BASE_SCALE))
        base_margin_y = int(round(ph * self.settings.search_radius_scale * DEFAULT_SEARCH_BASE_SCALE))
        motion_margin_x = int(round(abs(float(self.velocity[0])) * DEFAULT_SEARCH_MOTION_GAIN))
        motion_margin_y = int(round(abs(float(self.velocity[1])) * DEFAULT_SEARCH_MOTION_GAIN))
        margin_x = max(DEFAULT_SEARCH_MIN_MARGIN, base_margin_x + motion_margin_x)
        margin_y = max(DEFAULT_SEARCH_MIN_MARGIN, base_margin_y + motion_margin_y)

        if self._occluded_streak > 0:
            margin_x = int(round(margin_x * DEFAULT_OCCLUSION_SEARCH_GAIN))
            margin_y = int(round(margin_y * DEFAULT_OCCLUSION_SEARCH_GAIN))

        sx1 = max(0, px - margin_x)
        sy1 = max(0, py - margin_y)
        sx2 = min(frame_w, px + pw + margin_x)
        sy2 = min(frame_h, py + ph + margin_y)
        search_region = gray[sy1:sy2, sx1:sx2]
        if search_region.shape[0] < ph or search_region.shape[1] < pw:
            return None

        template_gray = cv2.resize(self.template_gray, (pw, ph), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(search_region, template_gray, cv2.TM_CCOEFF_NORMED)
        _, raw_score, _, best_loc = cv2.minMaxLoc(result)
        response_margin = _response_margin(result, best_loc, pw, ph)

        candidate_bbox = _clamp_bbox(
            (sx1 + int(best_loc[0]), sy1 + int(best_loc[1]), pw, ph),
            frame_w,
            frame_h,
        )

        x, y, w, h = candidate_bbox
        patch_gray = gray[y : y + h, x : x + w]
        if patch_gray.shape[:2] != (ph, pw):
            return None

        match_score = float(np.clip((raw_score + 1.0) / 2.0, 0.0, 1.0))
        jump = _jump_factor(self.bbox, candidate_bbox)
        spatial_penalty = float(np.exp(-0.35 * jump * jump))
        score = float(np.clip(match_score * spatial_penalty, 0.0, 1.0))

        return Candidate(
            bbox=candidate_bbox,
            score=score,
            jump_factor=jump,
            match_score=match_score,
            response_margin=response_margin,
        )

    def _set_template_from_frame(self, frame: np.ndarray, bbox: BBox) -> None:
        x, y, w, h = bbox
        patch = frame[y : y + h, x : x + w]
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        self.template_gray = patch_gray

    def _maybe_update_template(self, gray: np.ndarray, candidate: Candidate) -> None:
        if self.template_gray is None:
            return
        if (
            candidate.score < DEFAULT_TEMPLATE_UPDATE_MIN_CONFIDENCE
            or candidate.response_margin < DEFAULT_TEMPLATE_UPDATE_MIN_RESPONSE_MARGIN
            or candidate.jump_factor > DEFAULT_TEMPLATE_UPDATE_MAX_JUMP
        ):
            return
        self._track_frame_count += 1
        if self._track_frame_count % DEFAULT_TEMPLATE_UPDATE_INTERVAL != 0:
            return
        x, y, w, h = candidate.bbox
        patch = gray[y : y + h, x : x + w]
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            return
        th, tw = self.template_gray.shape[:2]
        resized = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_LINEAR)
        self.template_gray = cv2.addWeighted(
            self.template_gray, 1.0 - DEFAULT_TEMPLATE_UPDATE_ALPHA,
            resized, DEFAULT_TEMPLATE_UPDATE_ALPHA, 0,
        ).astype(np.uint8)

    def _update_velocity(self, previous_bbox: BBox, new_bbox: BBox) -> None:
        prev_cx = previous_bbox[0] + previous_bbox[2] / 2.0
        prev_cy = previous_bbox[1] + previous_bbox[3] / 2.0
        new_cx = new_bbox[0] + new_bbox[2] / 2.0
        new_cy = new_bbox[1] + new_bbox[3] / 2.0
        measured = np.array([new_cx - prev_cx, new_cy - prev_cy], dtype=np.float32)
        self.velocity = (
            DEFAULT_MOTION_PREDICTION_BLEND * self.velocity
            + (1.0 - DEFAULT_MOTION_PREDICTION_BLEND) * measured
        )

    def _state_from_observation(self, confidence: float, confidence_ema: float, jump_factor: float) -> str:
        if confidence < self.settings.lost_confidence_threshold and confidence_ema < self.settings.lost_confidence_threshold:
            return "Lost"
        if jump_factor > DEFAULT_MAX_RELATIVE_CENTER_JUMP:
            return "Uncertain"
        if self.prev_confidence is not None and confidence + 0.18 < self.prev_confidence:
            return "Uncertain"
        if confidence_ema < self.settings.uncertain_confidence_threshold:
            return "Uncertain"
        return "Tracking"

    def _is_candidate_acceptable(self, candidate: Candidate) -> bool:
        if candidate.match_score < DEFAULT_CANDIDATE_MATCH_THRESHOLD:
            return False
        if candidate.response_margin < DEFAULT_RESPONSE_MARGIN_THRESHOLD:
            return False
        if candidate.jump_factor > DEFAULT_ALLOWED_RECOVERY_JUMP:
            return False
        if candidate.jump_factor > DEFAULT_STRICT_JUMP_THRESHOLD:
            strong_recovery = (
                candidate.match_score >= DEFAULT_STRONG_MATCH_THRESHOLD
                and candidate.response_margin >= DEFAULT_STRONG_RESPONSE_MARGIN
            )
            if candidate.score < self.settings.uncertain_confidence_threshold and not strong_recovery:
                return False
        return True

    def _check_occlusion(self, candidate: Candidate) -> None:
        if self.confidence_ema is not None and self.confidence_ema > 0.5:
            if candidate.score < 0.3 and candidate.match_score < 0.4:
                self._occluded_streak += 1
            else:
                self._occluded_streak = 0
                self._motion_only_frames = 0
        elif self._occluded_streak > 0:
            self._occluded_streak = min(self._occluded_streak + 1, DEFAULT_MOTION_ONLY_FRAMES + 2)

    def _handle_occlusion(self, frame: np.ndarray, gray: np.ndarray, start: float, frame_h: int, frame_w: int) -> TrackingResult:
        if self._motion_only_frames < DEFAULT_MOTION_ONLY_FRAMES:
            self._motion_only_frames += 1
            self.velocity *= 0.9
            predicted = self._predict_bbox(frame_w, frame_h)
            self.bbox = predicted
            latency_ms = (perf_counter() - start) * 1000.0
            return TrackingResult(self.bbox, 0.0, "Occluded", latency_ms)
        self._occluded_streak = 0
        self._motion_only_frames = 0
        self.low_confidence_streak += 1
        state = "Lost" if self.low_confidence_streak >= DEFAULT_LOST_STREAK else "Uncertain"
        latency_ms = (perf_counter() - start) * 1000.0
        return TrackingResult(self.bbox, 0.0, state, latency_ms)


def _jump_factor(previous_bbox: BBox, candidate_bbox: BBox) -> float:
    px, py, pw, ph = previous_bbox
    cx, cy, cw, ch = candidate_bbox
    dist = float(
        np.linalg.norm(np.array([cx + cw / 2.0 - (px + pw / 2.0), cy + ch / 2.0 - (py + ph / 2.0)]))
    )
    return dist / max(1.0, 0.5 * (pw + ph))


def _clamp_bbox(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
    x, y, w, h = bbox
    x = max(0, min(int(x), frame_w - 1))
    y = max(0, min(int(y), frame_h - 1))
    w = max(1, min(int(w), frame_w - x))
    h = max(1, min(int(h), frame_h - y))
    return (x, y, w, h)


def _response_margin(result: np.ndarray, best_loc: tuple[int, int], template_w: int, template_h: int) -> float:
    if result.size <= 1:
        return 0.0

    peak = float(result[best_loc[1], best_loc[0]])
    masked = result.copy()
    radius_x = max(1, template_w // 4)
    radius_y = max(1, template_h // 4)
    x1 = max(0, best_loc[0] - radius_x)
    y1 = max(0, best_loc[1] - radius_y)
    x2 = min(masked.shape[1], best_loc[0] + radius_x + 1)
    y2 = min(masked.shape[0], best_loc[1] + radius_y + 1)
    masked[y1:y2, x1:x2] = -1.0
    second_peak = float(np.max(masked))
    peak_norm = float(np.clip((peak + 1.0) / 2.0, 0.0, 1.0))
    second_norm = float(np.clip((second_peak + 1.0) / 2.0, 0.0, 1.0))
    return max(0.0, peak_norm - second_norm)
