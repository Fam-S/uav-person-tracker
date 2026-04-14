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
    if not config_path.exists(): raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f: config = yaml.safe_load(f)
    return config

def make_parent_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def create_video_writer(output_path: str | Path, fps: float, frame_width: int, frame_height: int) -> cv2.VideoWriter:
    output_path = make_parent_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened(): raise RuntimeError(f"Failed to create video writer: {output_path}")
    return writer

class PointSelector:
    def __init__(self, frame):
        self.frame = frame
        self.point = None
        self.box_size = 100  
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.point is not None:
                if flags > 0: self.box_size = min(500, self.box_size + 15) 
                else: self.box_size = max(30, self.box_size - 15)    

    def run(self):
        window_name = "Click Target & Scroll to Resize"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while not self.done:
            display = self.frame.copy()
            if self.point is not None:
                px, py = self.point
                half = self.box_size // 2
                cv2.rectangle(display, (px - half, py - half), (px + half, py + half), (0, 255, 0), 2)
                cv2.putText(display, f"Size: {self.box_size} | Press ENTER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Click on the target...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 13 and self.point is not None:  
                self.done = True
            elif key == 27:  
                self.point = None
                self.done = True
                
        cv2.destroyWindow(window_name)
        return self.point


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Siamese tracker on video")
    parser.add_argument("--config", type=str, default="configs/tracker.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--video", type=str, default=None, help="Direct path to video file")
    args = parser.parse_args()

    config = load_config(args.config)

    video_path = args.video or config.get("video_path")
    output_path = config.get("output_path", "data/output/tracked_output.mp4")
    
    template_size = int(config.get("template_size", 127))
    search_size = int(config.get("search_size", 255))
    device = config.get("device")
    display_output = bool(config.get("display_output", True))
    save_output = bool(config.get("save_output", True))
    debug = bool(config.get("debug", False))
    confidence_threshold = float(config.get("confidence_threshold", 0.35))

    # FIX: Read context_amount directly from config for the new tracker logic
    context_amount = float(config.get("context_amount", 0.5))

    tracking_threshold = max(0.6, confidence_threshold)
    uncertain_threshold = min(0.3, confidence_threshold)

    if not video_path: raise ValueError("Must provide --video or set video_path in tracker.yaml")

    print(f"Loading model...")
    model, device = load_model(args.checkpoint, device=device)

    predictor = Predictor(model=model, device=device, template_size=template_size, search_size=search_size)
    
    # FIX: Pass context_amount instead of search_scale
    tracker = SiameseTrackerInference(
        predictor=predictor, 
        template_size=template_size, 
        search_size=search_size,
        context_amount=context_amount,
        tracking_threshold=tracking_threshold, 
        uncertain_threshold=uncertain_threshold,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_video_writer(output_path, fps, frame_width, frame_height) if save_output else None

    ret, first_frame = cap.read()
    if not ret: raise RuntimeError("Failed to read first frame.")

    print("Please click on the target...")
    selector = PointSelector(first_frame)
    selected_point = selector.run()

    if selected_point is None:
        cap.release()
        if writer: writer.release()
        print("Cancelled.")
        return

    px, py = selected_point
    half = selector.box_size // 2
    init_bbox = (float(px - half), float(py - half), float(selector.box_size), float(selector.box_size))

    print(f"Starting tracking on Box: {init_bbox}")
    init_result = tracker.initialize(first_frame, init_bbox)
    trail_points = [bbox_center(init_result.bbox)]

    first_vis = draw_tracking_overlay(frame=first_frame, bbox=init_result.bbox, score=init_result.score, state=init_result.state, search_bbox=init_result.search_bbox, grid_pos=init_result.grid_pos, trail=trail_points)

    if display_output: cv2.imshow("Tracking", first_vis)
    if writer: writer.write(first_vis)

    frame_index = 1

    while True:
        ret, frame = cap.read()
        if not ret: break

        result = tracker.track(frame)
        trail_points.append(bbox_center(result.bbox))
        if len(trail_points) > 50: trail_points = trail_points[-50:]

        vis_frame = draw_tracking_overlay(
            frame=frame, bbox=result.bbox, score=result.score, state=result.state,
            search_bbox=result.search_bbox if debug else None,
            grid_pos=result.grid_pos if debug else None,
            trail=trail_points,
        )

        if debug:
            print(f"[frame {frame_index:04d}] bbox=({result.bbox[0]:.1f}, {result.bbox[1]:.1f}, {result.bbox[2]:.1f}, {result.bbox[3]:.1f}) score={result.score:.3f} state={result.state}")

        if display_output:
            cv2.imshow("Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

        if writer: writer.write(vis_frame)
        frame_index += 1

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("[OK] Tracking finished.")
    if save_output: print(f"[OK] Video saved to: {output_path}")

if __name__ == "__main__":
    main()