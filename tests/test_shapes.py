import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.inference.load_model import load_model
from src.inference.predictor import Predictor
from src.inference.tracker import SiameseTrackerInference

model, device = load_model("checkpoints/best.pth")

predictor = Predictor(
    model=model,
    device=device,
    template_size=127,
    search_size=255,
)

tracker = SiameseTrackerInference(
    predictor=predictor,
    template_size=127,
    search_size=255,
)

frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

print(tracker.initialize(frame1, (200, 150, 80, 120)))
print(tracker.track(frame2))