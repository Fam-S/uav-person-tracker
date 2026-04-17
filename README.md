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

---

## Final Product Vision

The final product is intended to be a **desktop GUI application** for UAV-assisted person tracking, not just a model demo.

Selected product direction:

- **Main use case:** **Outdoor Sports Filming Assistant**
- **Secondary considered use case:** **UAV-assisted search and rescue support**

The application is intended to load live or recorded UAV video, let the user select one person, track that person in real time, and expose clear tracking states such as **Tracking**, **Uncertain**, and **Lost**.

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
* Config-driven system (`config.yaml`)
* Modular codebase
* Training pipeline with per-epoch checkpointing
* Inference pipeline with interactive ROI selection and visualization

---

## Documentation

- [`PROJECT_NOTES.md`](PROJECT_NOTES.md) - Extended project reasoning, dataset strategy, product vision, and selected use-case narrative.
- [`docs/public_dataset_notes.md`](docs/public_dataset_notes.md) - Notes on the competition dataset, person-tracking alignment, and recommended supplementary public datasets.
- [`docs/siamlight_components.md`](docs/siamlight_components.md) - Breakdown of the main components in a SiamLight-style tracker.
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
