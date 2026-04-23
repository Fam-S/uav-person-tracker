# UAV Person Tracker

## Overview

This project implements a **real-time UAV person-tracking system** for **CPU-only edge deployment** and a **desktop GUI application**.

The main tracker direction is now **SiamAPN++ + MobileOne-S2**: a UAV-oriented APN head paired with a re-parameterizable MobileOne backbone.

The system takes UAV video input, initializes a target in the first frame, and continuously tracks that person across subsequent frames.

Current implementation status:

- the repository is in transition before the primary tracker rewrite starts
- the GUI and backend abstraction are already in place
- the planned primary model implementation is documented before the code swap lands

Current project scope:

- **Inference** for online tracking
- **Training** for model learning and fine-tuning

## Constraints

- CPU only -> limited parallelism and latency-sensitive operations
- 30 FPS target -> about 33 ms per frame
- 720p input -> avoid full-frame processing
- Single-object tracking -> template can be reused, but search still runs every frame
- Limited memory/cache -> large models hurt real performance

## Primary Tracker Plan

The planned primary architecture is:

- **SiamAPN++** tracking head
- **MobileOne-S2** backbone
- dual-level feature extraction from MobileOne intermediate stages
- inference-time **re-parameterization** for fast deployment

Why this direction:

- literature-backed improvement over older anchor-based Siamese baselines on UAV tracking
- better fit for small aerial targets than a plain lightweight cross-correlation head
- still well inside the project's CPU and competition budget

Implementation priorities:

- swap the default SiamAPN++ AlexNet backbone for MobileOne-S2
- adjust APN neck channels to match the new feature dimensions
- keep ImageNet pretrained unfused MobileOne weights for training
- fuse the MobileOne backbone once before inference

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
Tracker Backend
   ├── Current app backend: OpenCV CSRT
   └── Planned primary backend: SiamAPN++ + MobileOne-S2
       ├── Template branch
       ├── Search branch
       ├── MobileOne-S2 dual-level features
       ├── APN fusion / prediction head
       └── Box scoring and selection
   ↓
Tracking State + Post-processing
   ├── Tracking / Uncertain / Lost
   ├── Confidence and smoothing
   └── Overlay + event output
Tracked Person Output
```

---

## Features

* Planned primary tracker: `SiamAPN++ + MobileOne-S2`
* CPU-oriented design with MobileOne re-parameterization planned for deployment
* **Desktop GUI** (`app/`) built with PySide6 — load video, select target, track in real time
* Current app backend: OpenCV CSRT, behind a swappable tracker interface during the transition
* Threaded rendering — worker thread runs the tracker; Qt main thread renders at 30 fps without blocking
* Frame downscaling before tracking (`track_max_width`) for speed without sacrificing display resolution
* Explicit tracker states: Tracking, Uncertain, Lost
* Bounding box overlay, confidence label, and trajectory trail
* Config-driven system (`config.yaml` for shared project settings, `app/app_config.yaml` for the GUI)
* Modular codebase with a stable `TrackerBackend` interface for easy model swapping
* Training pipeline with per-epoch checkpointing
* Inference pipeline with interactive ROI selection and visualization

---

## Documentation

- [`docs/SiamAPN++ + MobileOne-S2 — Implementation Plan.md`](docs/SiamAPN++ + MobileOne-S2 — Implementation Plan.md) - Primary tracker implementation plan and architecture decisions.
- [`docs/Phase 0 — CNN & Siamese Baselines.md`](docs/Phase 0 — CNN & Siamese Baselines.md) - Research-backed baseline comparison and rationale for the chosen direction.
- [`docs/references/public_dataset_notes.md`](docs/references/public_dataset_notes.md) - Notes on the competition dataset, person-tracking alignment, and recommended supplementary public datasets.
- [`docs/info-for-agents.md`](docs/info-for-agents.md) - Consolidated notes about dataset structure, rules, scoring, and constraints from the overlapping MTC-AIC4 context.
- [`docs/references/models_comparisons.md`](docs/references/models_comparisons.md) - Older model comparison notes kept as reference material.

---

## Project Structure

```
uav-person-tracker/
│
├── config.yaml                 # Combined config (model, train, inference)
├── requirements.txt            # pip dependencies
│
├── app/                        # Desktop GUI application
│   ├── main.py                 # PySide6 entry point
│   └── app_config.yaml         # GUI and tracker backend settings
│
├── train/                      # Training package
│   ├── __init__.py
│   ├── metrics.py              # IoU, prediction selection, batch metrics
│   └── run.py                  # Training loop entry point
│
├── models/                     # Tracker and backbone implementations
│   ├── backbone/               # Backbone modules
│   └── ...                     # Tracker-specific model code
│
├── data/                       # Data pipeline
│   ├── competition_data.py     # Competition manifest + annotation loading
│   ├── competition_siamese_dataset.py  # Template/search pair sampling
│   ├── competition_submission.py       # Submission CSV I/O
│   ├── competition_video.py    # Video frame iteration helpers
│   ├── crop_utils.py           # Shared crop/projection utilities
│   └── input/                  # Sample input videos
│
├── evaluation/                 # Evaluation utilities
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
uv sync                # core dependencies (includes opencv-contrib-python)
uv sync --group ui     # install the desktop UI dependencies (PySide6)
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
python train/run.py --config config.yaml
python -m app.main
```

---

## Usage

### Prepare Dataset

Point `train.dataset_root` at the competition raw-data directory containing:

- `metadata/contestant_manifest.json`
- sequence videos referenced by the manifest
- annotation files referenced by the manifest

### Run Training

```bash
# Using uv
uv run train --config config.yaml
uv run project-config set train.epochs 20

# Using python directly
python train/run.py --config config.yaml
python -m train.config_cli set train.epochs 20
```

Options:

* `--config <path>` — path to YAML config (default: `config.yaml`)

Config CLI:

* `uv run project-config show`
* `uv run project-config list-keys`
* `uv run project-config get train.batch_size`
* `uv run project-config set train.batch_size 16`

### Run the Desktop GUI

```bash
uv sync --group ui
uv run python -m app.main
```

Flow: Open Video → click target center → drag to resize → Start → Pause / Reset

---

## Configuration

Settings are split by ownership:

- [`config.yaml`](config.yaml) for shared project settings
- [`app/app_config.yaml`](app/app_config.yaml) for GUI-only settings

| Config file | Section | Key Settings |
|-------------|---------|-------------|
| `config.yaml` | `model` | backbone variant, feature channels, template/search size, pretrained flag |
| `config.yaml` | `train` | dataset root, batch size, epochs, learning rate, workers, per-epoch pair sampling, jitter |
| `config.yaml` | `infer` | checkpoint path, video path, confidence threshold, device |
| `config.yaml` | `tracking` | backend, checkpoint, thresholds, crop scales, `track_max_width`, trail length |
| `app/app_config.yaml` | `tracking` | app override layer for the GUI-specific tracker behavior |
| `app/app_config.yaml` | `video` | `target_fps` |
| `app/app_config.yaml` | `overlay` | show confidence, show trail |

---

## License

This project is for educational and research purposes.

---

## Authors

Fatma Sabry Mahmoud
GitHub: [https://github.com/Fam-S](https://github.com/Fam-S)
Abdulhamed Eid
GitHub: [https://github.com/Abdo-Eid](https://github.com/Abdo-Eid)
