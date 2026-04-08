# UAV Person Tracker (Siamese-Based)

## Overview

This project implements a **real-time UAV-based person tracking system** using a **Siamese network architecture with a MobileNetV3 backbone**.

The system takes UAV video input, initializes a target in the first frame, and continuously tracks that person across subsequent frames.

It supports both:

* **Inference (tracking)**
* **Training (model learning and fine-tuning)**

---

## System Architecture

```
UAV Video
   ↓
Frame Extractor
   ↓
Siamese Tracker
   ├── Template Branch (first frame target)
   ├── Feature Extractor (MobileNetV3)
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
* Lightweight MobileNetV3 backbone
* Manual target initialization (ROI selection)
* Real-time frame-by-frame tracking
* Config-driven system
* Modular codebase
* Training pipeline support
* Debug outputs (bounding boxes, crops)

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
⬜ Full tracker implementation (in progress)
⬜ Model training and evaluation

---

## Limitations

* Single object tracking only
* No re-detection mechanism yet
* No GUI (command-line only)
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
* Focus: rapid prototyping within 1 week

---

## License

This project is for educational and research purposes.

---

## Author

Fatma Sabry Mahmoud
GitHub: [https://github.com/Fam-S](https://github.com/Fam-S)
