from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from src.inference.load_model import load_model
from src.inference.predictor import Predictor
from src.inference.tracker import SiameseTrackerInference
from src.inference.visualize import draw_tracking_overlay, bbox_center


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Tracker config must be a YAML mapping/object.")

    return config


def make_parent_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def create_video_writer(
    output_path: str | Path,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    output_path = make_parent_dir(output_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Siamese tracker on video")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tracker.yaml",
        help="Path to tracker config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to trained checkpoint",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    video_path = config.get("video_path")
    output_path = config.get("output_path", "data/output/tracked_output.mp4")

    template_size = int(config.get("template_size", 127))
    search_size = int(config.get("search_size", 255))

    device = config.get("device")
    display_output = bool(config.get("display_output", True))
    save_output = bool(config.get("save_output", True))
    debug = bool(config.get("debug", False))

    confidence_threshold = float(config.get("confidence_threshold", 0.35))

    search_scale = float(config.get("search_scale", 2.0))
    if "context_amount" in config:
        context_amount = float(config.get("context_amount", 0.5))
        search_scale = max(2.0, 1.0 + (2.0 * context_amount))

    tracking_threshold = max(0.6, confidence_threshold)
    uncertain_threshold = min(0.3, confidence_threshold)

    if video_path is None:
        raise ValueError("tracker.yaml must contain 'video_path'")

    # ---- load model ----
    model, device = load_model(args.checkpoint, device=device)

    predictor = Predictor(
        model=model,
        device=device,
        template_size=template_size,
        search_size=search_size,
    )

    tracker = SiameseTrackerInference(
        predictor=predictor,
        template_size=template_size,
        search_size=search_size,
        search_scale=search_scale,
        tracking_threshold=tracking_threshold,
        uncertain_threshold=uncertain_threshold,
    )

    # ---- open video ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_output:
        writer = create_video_writer(
            output_path=output_path,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    # ---- read first frame ----
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        if writer is not None:
            writer.release()
        raise RuntimeError("Failed to read first frame from video.")

    # ---- select target ROI ----
    init_bbox = cv2.selectROI("Select Target", first_frame, False, False)
    cv2.destroyWindow("Select Target")

    if init_bbox is None or len(init_bbox) != 4:
        cap.release()
        if writer is not None:
            writer.release()
        raise RuntimeError("No valid ROI selected.")

    init_x, init_y, init_w, init_h = init_bbox
    if init_w <= 0 or init_h <= 0:
        cap.release()
        if writer is not None:
            writer.release()
        raise RuntimeError("Selected ROI has invalid size.")

    init_result = tracker.initialize(
        first_frame,
        (float(init_x), float(init_y), float(init_w), float(init_h)),
    )

    trail_points = [bbox_center(init_result.bbox)]

    first_vis = draw_tracking_overlay(
        frame=first_frame,
        bbox=init_result.bbox,
        score=init_result.score,
        state=init_result.state,
        search_bbox=init_result.search_bbox,
        grid_pos=init_result.grid_pos,
        trail=trail_points,
    )

    if display_output:
        cv2.imshow("Tracking", first_vis)

    if writer is not None:
        writer.write(first_vis)

    frame_index = 1

    # ---- main loop ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = tracker.track(frame)
        trail_points.append(bbox_center(result.bbox))

        if len(trail_points) > 50:
            trail_points = trail_points[-50:]

        vis_frame = draw_tracking_overlay(
            frame=frame,
            bbox=result.bbox,
            score=result.score,
            state=result.state,
            search_bbox=result.search_bbox if debug else None,
            grid_pos=result.grid_pos if debug else None,
            trail=trail_points,
        )

        if debug:
            print(
                f"[frame {frame_index:04d}] "
                f"bbox=({result.bbox[0]:.1f}, {result.bbox[1]:.1f}, "
                f"{result.bbox[2]:.1f}, {result.bbox[3]:.1f}) "
                f"score={result.score:.3f} state={result.state}"
            )

        if display_output:
            cv2.imshow("Tracking", vis_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        if writer is not None:
            writer.write(vis_frame)

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("[OK] Tracking finished.")
    if save_output:
        print(f"[OK] Output video saved to: {output_path}")


if __name__ == "__main__":
    main()