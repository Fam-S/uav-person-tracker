# UAV Person Tracker (Siamese-Based)

## Overview

This project implements a **real-time UAV person-tracking system** using a **Siamese architecture with a MobileOne backbone**.

The target deployment is **CPU-only edge hardware**, so the design prioritizes low latency, lightweight models, and practical real-time performance.

For the longer project reasoning, dataset strategy, and product vision, see [`PROJECT_NOTES.md`](PROJECT_NOTES.md).

The current primary model choice is:

- **Symmetric Siamese tracker with MobileOne-S0 on both branches**

The system takes UAV video input, initializes a target in the first frame, and continuously tracks that person across subsequent frames.

Current project scope:

- **Inference** for online tracking
- **Training** for model learning and fine-tuning

## Constraints

- CPU only -> limited parallelism and latency-sensitive operations
- 30 FPS target -> about 33 ms per frame
- 720p input -> avoid full-frame processing
- Single-object tracking -> template can be reused, but search still runs every frame
- Limited memory/cache -> large models hurt real performance

---

## Dataset and Training Strategy

This project overlaps with the **MTC-AIC4 competition**, and uses that overlap as a strong starting point for efficient aerial single-object tracking.

Short version:

- **Stage 1:** train on the competition-style aerial tracking dataset for generic tracking behavior
- **Stage 2:** fine-tune on **VisDrone** for stronger UAV person-tracking performance
- **Stage 1 model:** suitable for a competition submission
- **Stage 2 model:** intended for the college GUI application

See [`PROJECT_NOTES.md`](PROJECT_NOTES.md) for the full reasoning and dataset notes.

---

## Final Product Vision

The final product is intended to be a **desktop GUI application** for UAV-assisted person tracking, not just a model demo.

Selected product direction:

- **Main use case:** **Outdoor Sports Filming Assistant**
- **Secondary considered use case:** **UAV-assisted search and rescue support**

The application is intended to load live or recorded UAV video, let the user select one person, track that person in real time, and expose clear tracking states such as **Tracking**, **Uncertain**, and **Lost**.

See [`PROJECT_NOTES.md`](PROJECT_NOTES.md) for the full product vision, scenario, narrative use case, and benefits.

---

## System Architecture

```
UAV Video
   ↓
Frame Extractor
   ↓
Siamese Tracker
   ├── Template Branch (first frame target)
   ├── Shared Feature Extractor (MobileOne-S0)
   ├── Search Branch (current frame)
   ├── Cross-Correlation
   ├── Response Map
   └── Bounding Box Head
   ↓
Post-Processing (smoothing + scale adaptation)
   ↓
Tracked Person Output
```

---

## Features

* Siamese-based visual tracking
* Lightweight MobileOne backbone
* Primary CPU-only design uses symmetric MobileOne-S0 branches
* Designed for edge deployment
* Manual target initialization (ROI selection)
* Real-time frame-by-frame tracking
* Config-driven system
* Modular codebase
* Training pipeline support
* Debug outputs (bounding boxes, crops)

---

## Documentation

- [`PROJECT_NOTES.md`](PROJECT_NOTES.md) - Extended project reasoning, dataset strategy, product vision, and selected use-case narrative.
- [`docs/models_comparisons.md`](docs/models_comparisons.md) - Backbone and tracker comparisons, including MobileOne, ShuffleNetV2, MobileNetV3, and alternative tracking directions.
- [`docs/public_dataset_notes.md`](docs/public_dataset_notes.md) - Notes on the competition dataset, person-tracking alignment, and recommended supplementary public datasets.
- [`docs/future_improvements.md`](docs/future_improvements.md) - Planned improvements for the primary MobileOne-S0 Siamese experiment.
- [`docs/future_experiments.md`](docs/future_experiments.md) - Future experiment ideas for long-term tracking, occlusion handling, and recovery logic.
- [`docs/siamese_tracking_occlusion_notes.md`](docs/siamese_tracking_occlusion_notes.md) - Explanation of Siamese tracking limits under occlusion and long-term target loss.
- [`docs/siamlight_components.md`](docs/siamlight_components.md) - Breakdown of the main components in a SiamLight-style tracker.
- [`docs/info-for-agents.md`](docs/info-for-agents.md) - Consolidated notes about dataset structure, rules, scoring, and constraints from the overlapping MTC-AIC4 context.
- [`docs/veo_test_video_prompts.md`](docs/veo_test_video_prompts.md) - Veo 3.1 prompts for generating realistic UAV tracking test videos, including stress-test cases.

---

## Project Structure

```
uav-person-tracker/
│
├── main.py                # Inference entry point
├── train.py               # Training entry point
├── validate.py            # Validation script
├── demo.py                # Demo runner
│
├── configs/               # Configuration files
├── data/                  # Input/output and datasets
├── models/                # Backbone + tracker modules
├── src/                   # Core pipeline (video, preprocessing)
├── training/              # Training pipeline
├── evaluation/            # Metrics and evaluation
├── debug/                 # Debug outputs
├── checkpoints/           # Saved models
└── logs/                  # Training logs
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Fam-S/uav-person-tracker.git
cd uav-person-tracker
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Inference (Tracking)

Place a video file at:

```
data/input/sample.mp4
```

Run:

```bash
python main.py --config configs\tracker.yaml
```

Steps:

1. First frame opens
2. Select the target person using mouse
3. Press Enter to confirm
4. Bounding box is saved in `debug/`

---

### Run Training

```bash
python train.py --config configs\train.yaml
```

This will:

* Load dataset
* Generate template-search pairs
* Train the tracking model
* Save checkpoints

---

### Run Validation

```bash
python validate.py --checkpoint checkpoints\best.pth
```

---

## Configuration

### tracker.yaml

Controls inference behavior:

* video path
* model parameters
* smoothing
* scale adaptation

### train.yaml

Controls training:

* dataset paths
* batch size
* learning rate
* epochs

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* YAML configuration

---

## Current Status

✔ Repo structure completed
✔ Inference pipeline initialized
✔ Training pipeline skeleton ready
✔ Primary CPU-only backbone selected: MobileOne-S0
⬜ Full tracker implementation in progress
⬜ Model training and evaluation in progress

---

## Limitations

* Single object tracking only
* No re-detection mechanism yet
* GUI application not implemented yet
* Training pipeline is basic (prototype level)

---

## Future Work

* Add re-detection module
* Improve tracking robustness under occlusion
* Multi-object tracking support
* Deploy on UAV edge devices
* Optimize inference speed (TensorRT / ONNX)

---

## Team

* 2-person development team
* Focus: rapid prototyping within a one-week timeline

---

## License

This project is for educational and research purposes.

---

## Authors

Fatma Sabry Mahmoud
GitHub: [https://github.com/Fam-S](https://github.com/Fam-S)
Abdulhamed Eid
GitHub: [https://github.com/Abdo-Eid](https://github.com/Abdo-Eid)
