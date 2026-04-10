import argparse
import yaml
import cv2
from src.preprocess import get_template, get_search_region, preprocess_image


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="UAV Person Tracker Inference")
    parser.add_argument("--config", type=str, default="configs\\tracker.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Inference config loaded successfully:")
    for k, v in config.items():
        print(f"{k}: {v}")

    video_path = config["video_path"]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(" Error: Could not open video")
        return

    print("Video opened successfully")

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return

    bbox = cv2.selectROI("Select Target", frame, False, False)
    print("Selected BBox:", bbox)

    cv2.destroyWindow("Select Target")

    template = get_template(frame, bbox)
    template = preprocess_image(template)

    print("Template shape:", template.shape)

    cv2.imshow("Template", template)
    cv2.waitKey(1000) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        search = get_search_region(frame, bbox)
        search = preprocess_image(search)

        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()