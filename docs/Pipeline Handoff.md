# Pipeline Handoff

## Overview

SiamAPN-style siamese tracker with Apple's MobileOne-S2 backbone. Trains on competition UAV data and generates Kaggle submission CSVs.

## Setup on Kaggle

1. Create a new Kaggle notebook with **GPU** accelerator
2. Add the competition dataset (it will be available at `/kaggle/input/...`)
   Use `/kaggle/input/datasets/abdulhamedeid/mtc-aic4-uav-dataset` as the dataset root for `train.dataset_root` and evaluation `--raw-root`.
3. Add this repo as a dataset or clone it:
   ```bash
   !git clone --branch <branch-name> https://github.com/Fam-S/uav-person-tracker.git /kaggle/working/uav-person-tracker
   %cd /kaggle/working/uav-person-tracker
   ```
4. Install uv and dependencies:
   ```bash
   !curl -LsSf https://astral.sh/uv/install.sh | sh
   %env UV_LINK_MODE=copy
   !uv sync
   ```
5. Download MobileOne-S2 pretrained weights:
   ```bash
   !mkdir -p checkpoints
   !curl -L -o checkpoints/mobileone_s2_unfused.pth.tar https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s2_unfused.pth.tar
   ```
6. Choose one config workflow before training:

   **Option A — persistent changes with the config CLI**
   ```bash
   !uv run project-config set train.dataset_root /kaggle/input/competition-name
   !uv run project-config set train.epochs 20
   !uv run project-config set train.batch_size 16
   ```

   **Option B — temporary run-only overrides**
   ```bash
   !uv run train --config config.yaml \
     --override train.dataset_root=/kaggle/input/competition-name \
     --override train.epochs=20 \
     --override train.batch_size=16
   ```

   - `project-config set` writes the new values into `config.yaml`.
   - `--override KEY=VALUE` changes only the resolved config for that run and does **not** modify `config.yaml` on disk.
7. For submission output, update `--output` to write to `/kaggle/working/` so you can download it.

## Train

```bash
uv run train --config config.yaml
```

Or keep `config.yaml` unchanged and override specific values just for one run:

```bash
uv run train --config config.yaml --override train.epochs=5 --override train.device=cpu
```

- Reads 255 train sequences from `data/raw/`
- Samples 2048 template/search pairs per epoch with translation + scale jitter
- Saves every epoch to `checkpoints/epoch_NNN.pth`
- Saves best (lowest loss) to `checkpoints/best.pth`
- Best epochs are marked with `*` in logs

Key config (`config.yaml` → `train`):

| Setting | Default | Notes |
|---------|---------|-------|
| `dataset_root` | `data/raw` | Must contain `metadata/contestant_manifest.json` |
| `epochs` | `20` | |
| `batch_size` | `8` | Reduce if OOM |
| `train_samples_per_epoch` | `2048` | Number of pairs sampled per epoch |
| `translation_jitter` | `0.15` | Random shift on search crop center |
| `scale_jitter` | `0.1` | Random scale change on search crop |
| `learning_rate` | `0.0001` | Fixed (no scheduler) |

## Generate Submission

```bash
uv run python -m evaluation.basic_eval --raw-root data/raw --output evaluation/submissions/siamapn_submission.csv
```

- Loads `checkpoints/best.pth`
- Runs `SiamAPNBackend` over all 89 public_lb sequences
- Writes submission CSV matching Kaggle format (`id,x,y,w,h`)
- Uses `config.yaml` → `tracking.backend: siamapn`

## Switch Between Backends

In `config.yaml`:

```yaml
tracking:
  backend: siamapn   # neural network tracker
  checkpoint: checkpoints/best.pth
```

```yaml
tracking:
  backend: csrt      # OpenCV CSRT (no GPU needed)
  checkpoint: null
```

## Architecture

```
models/backbone/_mobileone.py    Apple's MobileOne (self-contained, no external/ dependency)
models/backbone/mobileone.py     MobileOneS2Backbone wrapper (stage2=256ch, stage3=640ch)
models/siamapn.py                SiamAPNppMobileOne (dual-level correlation + regression head)
models/losses.py                 SmoothL1 regression loss
data/competition_siamese_dataset.py  Template/search pair sampling from competition data
data/competition_data.py         Manifest loading, sequence records
data/crop_utils.py               Shared crop extraction and box projection helpers
data/competition_submission.py   Submission CSV I/O
app/tracking.py                  SiamAPNBackend, CSRTBackend, MockBackend
train/run.py                     Training loop with best checkpoint saving
evaluation/basic_eval.py         Public LB evaluation + submission writer
```

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | All settings: model, training, inference, tracking |
| `config.py` | Config dataclasses and loader |
| `tests/test_architecture.py` | Backbone shapes, forward pass, loss verification |
| `notebooks/colab-varification.ipynb` | Colab smoke test notebook |
