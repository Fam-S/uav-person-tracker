# Layer 4 — Model Design

## Goal
Improve tracking accuracy and training stability without changing the overall Siamese architecture.

## Changes

### 1. LR Scheduler — Cosine Annealing
Replace the fixed learning rate with `torch.optim.lr_scheduler.CosineAnnealingLR`.
- LR decays smoothly from `learning_rate` → 0 over the total number of epochs.
- Add `scheduler.step()` after each epoch in the training loop.
- Add `scheduler_state_dict` to checkpoint save/load.

### 2. Focal Loss for Classification
Replace `BCEWithLogitsLoss` with Focal Loss in `models/losses.py`.
- Focal Loss downweights easy negatives, forcing the model to focus on hard anchors.
- Adds two hyperparams: `focal_alpha` (default 0.25) and `focal_gamma` (default 2.0).
- Add both to `config.yaml` under `train:`.

### 3. Channel Reduction Before Correlation
In `models/siamese.py`, add a 1×1 conv projection before the cross-correlation step:
- Project from `feature_channels` (96) → `corr_channels` (32) before correlation.
- Run depthwise cross-correlation on the reduced channels.
- Project back to `feature_channels` (96) before the head.
- Result: correlation map is 32-channel not 96-channel → ~3× cheaper, often better.
- Add `corr_channels` (default 32) to `model:` section of `config.yaml`.

### 4. Gradual Backbone Unfreezing
Add a `unfreeze_at_epoch` key to `config.yaml` under `train:`.
- If set, unfreeze the last N backbone blocks starting at that epoch.
- Implemented by checking `epoch == unfreeze_at_epoch` inside the training loop.
- Lets the backbone adapt after the head has converged.

## Config Keys Added

```yaml
model:
  corr_channels: 32         # channel dim inside cross-correlation

train:
  focal_alpha: 0.25
  focal_gamma: 2.0
  unfreeze_at_epoch: null   # set to an epoch number to enable gradual unfreezing
```

## Files to Modify
- `models/losses.py` — add `FocalLoss`, swap into `SiameseLoss`
- `models/siamese.py` — add channel reduction projection around correlation
- `train.py` — add scheduler, gradual unfreeze logic
- `config.yaml` — add new keys
