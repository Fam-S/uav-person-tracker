# UAV Person Tracker (Siamese-Based)

## Overview

This project implements a **real-time UAV person-tracking system** using a **Siamese tracking architecture**.

The target deployment is **CPU-only edge hardware**, so the design prioritizes low latency, lightweight models, and practical real-time performance.

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

This project overlaps with the **MTC-AIC4 competition**, but the broader goal is a practical **UAV person-tracking application**.

Recommended training path:

- **Stage 1:** train on the current aerial single-object tracking dataset for generic UAV tracking behavior
- **Stage 2:** fine-tune on **VisDrone** for stronger person-focused performance

Why this split makes sense:

- the competition-style dataset is strong for generic aerial single-object tracking
- it is not person-only, so it should not be the only source for a person-tracking application
- **VisDrone** is the best first supplementary dataset because it provides real UAV imagery with strong pedestrian presence, clutter, small targets, and occlusion

Intended use of each stage:

- **Stage 1 model:** suitable for a competition-oriented checkpoint
- **Stage 2 model:** intended for the college GUI application

If long occlusion and recovery become more important later, **UAV20L-style** long sequences are a good addition for evaluation or supplementary training.

---

## Final Product Vision

The final product is intended to be a **desktop GUI application** for UAV-assisted person tracking, not just a model demo.

The application should:

- load live drone feed or recorded video
- let the user select **one person** in the first frame
- track that person in real time
- display a clear tracking state: **Tracking**, **Uncertain**, or **Lost**
- overlay the predicted bounding box, confidence score, and short trajectory trail
- save tracked video, frame-by-frame predicted boxes, and target lost or recovered events

Suggested interface layout:

- a main video panel
- a side control panel for loading video, selecting the target, and starting or pausing tracking
- a status panel for FPS, confidence, latency, and current tracker state
- an event log or timeline for target loss and recovery

### Selected Use Case

The main selected use case is an **Outdoor Sports Filming Assistant**.

In that scenario, an operator selects one athlete at the start of a session, and the system helps keep that person visible and consistently framed during outdoor filming. The tracker state and confidence help the operator react early when the target becomes hard to see instead of silently drifting.

---

## System Architecture

```
UAV Video
   ↓
Frame Extractor
   ↓
Siamese Tracker
   ├── Template Branch (first frame target)
   ├── Shared Feature Extractor
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
* Lightweight Siamese tracking pipeline
* Target CPU-only design uses symmetric MobileOne-S0 branches
* Designed for edge deployment
* Manual target initialization (ROI selection)
* Real-time frame-by-frame tracking
* Config-driven system (`config.yaml`)
* Modular codebase
* Training pipeline with per-epoch checkpointing
* Inference pipeline with interactive ROI selection and visualization

---

## Documentation

- [`docs/public_dataset_notes.md`](docs/public_dataset_notes.md) - Notes on the competition dataset, person-tracking alignment, and recommended supplementary public datasets.
- [`docs/info-for-agents.md`](docs/info-for-agents.md) - Consolidated notes about dataset structure, rules, scoring, and constraints from the overlapping MTC-AIC4 context.

---

## Project Structure

```
uav-person-tracker/
│
├── config.yaml                 # Combined config (model, train, inference)
├── requirements.txt            # pip dependencies
│
├── train/                      # Training package
│   ├── __init__.py
│   ├── metrics.py              # IoU, prediction selection, batch metrics
│   └── run.py                  # SiameseTrainer + training loop entry point
│
├── inference/                  # Inference package
│   ├── load_model.py           # Load checkpoint → SiameseTracker
│   ├── predictor.py            # numpy image pair → PredictionResult
│   ├── tracker.py              # Stateful frame-by-frame tracker
│   ├── video_runner.py         # Full video pipeline with interactive ROI
│   └── visualize.py            # Bounding box + trail overlay drawing
│
├── models/                     # Model architecture
│   ├── backbone/
│   │   └── mobilenetv3.py      # MobileNetV3-Small feature extractor
│   ├── siamese.py              # DepthwiseCrossCorrelation + SiameseHead + SiameseTracker
│   └── losses.py               # SiameseLoss + build_targets
│
├── data/                       # Data pipeline
│   ├── dataset.py              # UAV123SiameseDataset
│   ├── preprocess.py           # Raw UAV123 → manifest/cache/splits
│   └── input/                  # Sample input videos
│
├── evaluation/                 # Evaluation (placeholder)
├── tests/                      # Test suite
├── docs/                       # Documentation
└── notebooks/                  # Jupyter notebooks
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Fam-S/uav-person-tracker.git
cd uav-person-tracker
```

### 2. Install dependencies

Using **uv** (recommended):

```bash
uv sync                # core dependencies
uv sync --group dev    # include pytest
```

Using **pip**:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

---

## Quick Start (without uv)

If you prefer pip or already have dependencies installed globally:

```bash
pip install -r requirements.txt
python data/preprocess.py --dataset-root <path-to-dataset> --output-dir data/processed
python train/run.py --config config.yaml
python inference/video_runner.py --checkpoint checkpoints/best.pth --video <path-to-video>
```

---

## Usage

### Preprocess Dataset

```bash
# Using uv
uv run preprocess --dataset-root <path-to-dataset> --output-dir data/processed

# Using python directly
python data/preprocess.py --dataset-root <path-to-dataset> --output-dir data/processed
```

### Run Training

```bash
# Using uv
uv run train --config config.yaml

# Using python directly
python train/run.py --config config.yaml
```

Options:

* `--config <path>` — path to YAML config (default: `config.yaml`)
* `--resume <path>` — resume from a checkpoint (e.g. `checkpoints/epoch_005.pth`)

### Run Inference (Tracking)

```bash
# Using uv
uv run infer --checkpoint checkpoints/best.pth --video <path-to-video>

# Using python directly
python inference/video_runner.py --checkpoint checkpoints/best.pth --video <path-to-video>
```

---

## Configuration

All settings live in a single [`config.yaml`](config.yaml):

| Section | Key Settings |
|---------|-------------|
| `model` | backbone variant, feature channels, template/search size, pretrained flag |
| `train` | dataset paths, batch size, learning rate, epochs, loss weights, backbone freeze |
| `infer` | checkpoint path, video path, confidence threshold, device |

---

## License

This project is for educational and research purposes.

---

## Authors

Fatma Sabry Mahmoud
GitHub: [https://github.com/Fam-S](https://github.com/Fam-S)
Abdulhamed Eid
GitHub: [https://github.com/Abdo-Eid](https://github.com/Abdo-Eid)
