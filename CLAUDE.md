# UAV Person Tracker — Claude Code Guide

## Project Layout

```
models/backbone/mobilenetv3.py   MobileNetV3 feature extractor
models/siamese.py                DepthwiseCrossCorrelation + SiameseHead + SiameseTracker
models/losses.py                 SiameseLoss + build_targets (Gaussian heatmap cls + SmoothL1 reg)
data/dataset.py                  UAV123DatasetLoader + UAV123PairSampler + UAV123SiameseDataset
data/preprocess.py               One-time preprocessing: raw UAV123 → manifest/cache/splits
inference/load_model.py          Load checkpoint → SiameseTracker (eval mode)
inference/predictor.py           Predictor: numpy image pair → PredictionResult
inference/tracker.py             SiameseTrackerInference: stateful frame-by-frame tracker
inference/video_runner.py        Full video pipeline with interactive target selection
train.py                         SiameseTrainer + compute_batch_metrics + training loop (all-in-one)
infer.py                         Thin shim → inference.video_runner.main()
```

## Common Commands

```bash
# Install (first time)
uv sync
uv sync --group dev   # include pytest

# Run training
uv run train --config configs/train.yaml

# Run inference on video
uv run infer --checkpoint checkpoints/best.pth --video data/input/sample.mp4

# Preprocess UAV123 dataset
uv run preprocess --dataset-root data/raw/UAV123 --output-dir data/processed

# Tests (no data or checkpoint required)
uv run pytest tests/test_architecture.py -v
```

## Key Import Paths

```python
from models.backbone import MobileNetV3Backbone
from models.siamese import SiameseTracker
from models.losses import SiameseLoss
from data.dataset import UAV123SiameseDataset
from inference.load_model import load_model
from train import compute_batch_metrics, SiameseTrainer, EpochStats
```

## Architecture Notes

- **Backbone**: MobileNetV3-Small, pretrained ImageNet, frozen by default. Output: `[B, 96, H/32, W/32]`
- **Correlation**: Depthwise cross-correlation (grouped conv trick). Template `[B,96,4,4]` × Search `[B,96,8,8]` → `[B,96,5,5]`
- **Head**: Dual branch — cls `[B,1,5,5]` (Gaussian heatmap target) + reg `[B,4,5,5]` (offset + log-scale)
- **Template size**: 127×127, **Search size**: 255×255
- **Loss**: BCEWithLogitsLoss (cls) + masked SmoothL1 (reg) on positive anchor only

## Config Files

- `configs/train.yaml` — dataset paths, lr, batch size, epochs, backbone settings
- `configs/tracker.yaml` — video path, output path, confidence thresholds, display options

## Data Pipeline

1. `uv run preprocess` → creates `data/processed/{manifest.json, annotations_cache.json, splits/}`
2. `UAV123SiameseDataset` reads from processed cache for fast loading
3. Template crop: scale×1.0, context=0.5, no jitter
4. Search crop: scale×2.0, context=0.5, translation_jitter=0.15, scale_jitter=0.1
