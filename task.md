# Tracking/Training Issues Identified

## Summary

The main problem appears to be in the **current rewritten implementation**, not in the original SiamAPN++ code under `external/`, and not yet in a proven MobileOne backbone bug. The strongest evidence points to a mismatch between how the model predicts boxes and how training targets are represented.

Current decision update:

- The project should target a faithful/full SiamAPN++ implementation, not a simplified direct-regression tracker.
- The partial center-head work is no longer the final direction; it was useful for diagnosing the failure but should be replaced by the original-style APN, cls/loc heads, target generation, loss terms, and tracker decode.
- The original implementation under `external/SiamAPN/SiamAPN++` is now the reference contract for the active implementation.
- Full-port implementation decision: copy/adapt the original SiamAPN++ logic into the active codebase instead of importing directly from `external/` at runtime.
- `external/` should remain unchanged as the upstream/reference copy.
- The active project should switch to the original SiamAPN++ geometry: template size `127`, search size `287`, output size `21`, anchor stride `8`.
- Earlier normalized-box and center-head work is superseded by the active full SiamAPN++ port. Keep those notes only as diagnostic history, not as the current implementation target.

Reason for copying/adapting instead of importing from `external/` directly:

- The original code hardcodes `.cuda()` in many model/loss/target paths.
- The original code relies on global `cfg` from `yacs`, while the active project uses dataclass/YAML config.
- The original package imports are tied to the `pysot` package layout under `external/`.
- The app/evaluation/training CLIs need clean integration with this repo's active modules and commands.
- Keeping `external/` unchanged makes it possible to compare our port against the original source.

---

## Issue 1: Regression target/output scale mismatch

### What is happening

The current model predicts a single 4-value box directly from a pooled regression head:

- `models/siamapn.py:64-70`
- `models/siamapn.py:95-96`

Training compares that output directly against `search_bbox_xywh` using plain Smooth L1 loss:

- `models/losses.py:21-39`

The dataset target boxes are generated in **search-crop pixel coordinates**:

- `data/competition_siamese_dataset.py:226-241`
- `data/crop_utils.py:78-116`

This means the model is learning against targets roughly on a `0..255` pixel scale, while the raw prediction head starts near very small values.

### Evidence

Observed target statistics during inspection were approximately:

- target mean: `[92.8, 91.2, 70.3, 72.5]`
- target min/max ranged roughly from `0` up to `247`

Observed model predictions were approximately:

- initial prediction mean: `[-0.47, 1.14, 0.56, 1.69]`
- checkpoint prediction mean still remained very small compared with targets

This matches the training log behavior where the loss stays around `79` and does not meaningfully decrease.

### Why it causes the tracker to stick to a point

During inference, the predicted crop-space box is projected back into frame coordinates here:

- `app/tracking.py:260-269`

If the model outputs an almost constant tiny crop-space box every frame, the tracker keeps projecting nearly the same location back into the image. Visually, that looks like the tracker is trying to lock onto one fixed point.

### Notes

- This is the highest-confidence issue.
- The likely fix is to use a normalized or center-based box parameterization instead of raw crop-pixel `x, y, w, h` regression.

---

## Issue 2: Current active model is not a faithful SiamAPN++ implementation

### What is happening

The original SiamAPN++ implementation in `external/` uses richer tracking logic with classification/score branches and localization decoding:

- `external/SiamAPN/SiamAPN++/pysot/tracker/adsiamapn_tracker.py:129-190`
- `external/SiamAPN/SiamAPN++/pysot/datasets/anchortarget_adapn.py:29-93`

The active rewritten model is much simpler:

- dual-level backbone features
- cross-correlation
- one fused feature map
- one pooled regression head producing a single box

Relevant code:

- `models/siamapn.py:56-70`
- `models/siamapn.py:82-99`

### Why this matters

The current implementation is only **inspired by** SiamAPN++, not equivalent to it. It removes key mechanisms that usually help Siamese trackers localize the target robustly, such as:

- classification/objectness scoring
- structured localization maps
- anchor/grid-style decoding
- score-based candidate selection and penalties

### Notes

- This does **not** mean the original repo in `external/` is wrong.
- It means the rewritten implementation does not currently preserve the original training/inference design.
- Even if Issue 1 is fixed, this simplification may still limit tracking quality.

---

## Issue 3: No confidence/objectness branch in the active training loop

### What is happening

The current training loop only optimizes regression loss:

- `train/run.py:95-124`
- `models/losses.py:38-43`

There is no classification loss, no positive/negative sampling logic, and no confidence branch in the active model/loss path.

### Why this matters

Without an objectness/classification signal, the model has little incentive to learn **where the target is most likely to be** versus background. A single global pooled regression head can collapse toward an average prediction that reduces loss but does not track well.

### Notes

- This increases the chance of trivial solutions such as predicting nearly constant boxes.
- This issue is lower confidence than Issue 1, but it is still a meaningful design weakness.

---

## Issue 4: Inference currently trusts the raw regressed box directly

### What is happening

Inference uses the regression output directly after `softplus` on width/height:

- `app/tracking.py:257-268`
- `models/siamapn.py:95-96`

There is no candidate ranking, no response-map peak selection, and no score-guided fallback logic like in the original SiamAPN++ tracker.

### Why this matters

If the regressor is weak or biased, inference has no second mechanism to reject bad localization. The tracker simply accepts the raw box and updates state.

### Notes

- This makes failure more visible once regression becomes unstable or biased.
- It is part of the reason the tracker can drift or stay fixed.

---

## What does **not** currently look like the main issue

### MobileOne backbone bug

No strong evidence yet shows that MobileOne itself is broken. The code runs, the tensors flow, and the failure pattern looks more like a training-target formulation problem than a backbone malfunction.

Relevant backbone entry:

- `models/backbone/mobileone.py`

### Basic merge/shape failure between SiamAPN++ and MobileOne

No strong evidence yet shows a direct feature-shape or merge bug causing runtime corruption. If this were the dominant issue, more obvious failures such as shape errors, NaNs, or immediate instability would be expected.

---

## Practical conclusion

### Most likely root cause order

1. **Regression target/output mismatch in the current implementation**
2. **Architecture simplification relative to original SiamAPN++**
3. **Missing classification/objectness/localization-map logic**
4. **Possible merge-specific issue** only if problems remain after fixing the above

### Recommended technical direction

The first fix should be to redesign the box prediction target and decoder so that training and inference use a normalized, consistent representation. After that, decide whether to:

- keep the simplified tracker and make it stable, or
- move closer to the original SiamAPN++ head and target design while keeping MobileOne as backbone.

---

## Plan Checklist

### Step 1: Inspect and document the current box parameterization path

Checklist:

- [x] Trace how target boxes are generated in the dataset
- [x] Trace how predictions are produced by the model head
- [x] Trace how predictions are decoded back to frame coordinates during inference
- [x] Confirm that training targets and inference decoding use the same box meaning

Expected result after completion:

The full regression path is explicitly verified end to end, and any mismatch in box representation is clearly identified in code.

Step 1 notes:

- Dataset targets are produced in `data/competition_siamese_dataset.py` by `project_box_to_crop(...)`, which converts the ground-truth search box into `search_bbox_xywh` in search-crop pixel coordinates.
- `project_box_to_crop(...)` in `data/crop_utils.py` computes the crop frame using the template/reference box, then projects the search box into the crop and clips it to `[0, out_size]`.
- The active model in `models/siamapn.py` predicts one raw 4-value box from pooled fused features; only width/height go through `softplus`.
- Inference in `app/tracking.py` feeds that raw prediction directly into `project_box_from_crop(...)`, which interprets it as crop-space `x, y, w, h` and maps it back into frame coordinates.
- The training and inference paths agree on the meaning of the box coordinates, but the representation is still raw crop-pixel `x, y, w, h`, which is the scale problem identified in Issue 1.

### Step 2: Redesign the regression target representation

Checklist:

- [x] Choose a normalized box representation for training
- [x] Prefer center-based or otherwise scale-stable parameterization over raw crop-pixel `x, y, w, h`
- [x] Define the exact encode/decode formulas to be used in both training and inference
- [x] Ensure the representation is numerically stable for small and large boxes

Expected result after completion:

There is one consistent box encoding design that the dataset, model loss, and tracker all follow.

Step 2 notes:

- The active training target now uses normalized center-based crop boxes: `search_bbox_cxcywh`.
- `data/competition_siamese_dataset.py` now encodes the crop box as `[cx, cy, w, h] / search_size` using `project_box_to_crop_center_norm(...)`.
- `models/siamapn.py` now applies `sigmoid` to the box head so the prediction stays in the same normalized range.
- `app/tracking.py` now decodes normalized center-based predictions with `project_box_from_crop_center_norm(...)` before projecting back to frame coordinates.
- `tests/test_crop_utils.py` and `tests/test_shapes.py` now enforce the `[0, 1]` range and the new key name.

### Step 3: Update dataset target generation

Checklist:

- [x] Change dataset targets to emit the new encoded box representation
- [x] Keep the crop generation logic unchanged unless a separate bug is found
- [x] Verify target values are in a sensible numeric range
- [x] Update or add tests for the new target contract

Expected result after completion:

Training batches contain encoded targets that are properly scaled and suitable for optimization.

Step 3 notes:

- `CompetitionSiameseDataset` now returns `search_bbox_cxcywh` instead of raw pixel-space `search_bbox_xywh`.
- The crop projection logic stays the same; only the target encoding changed.

### Step 4: Update model output and loss to match the new target format

Checklist:

- [x] Make the regression head output values in the new representation
- [x] Remove any output transform that conflicts with the new encoding
- [x] Update the regression loss to compare like-for-like values
- [x] Verify initial predictions and targets are on comparable scales

Expected result after completion:

The model trains against targets in the same numeric space as its outputs, so the loss should become meaningful instead of being dominated by scale mismatch.

Step 4 notes:

- `SiamAPNppMobileOne` now emits a sigmoid-bounded `[cx, cy, w, h]` box.
- The loss still uses Smooth L1, but now it compares normalized values instead of raw crop pixels.

### Step 5: Update inference decoding

Checklist:

- [x] Decode the new predicted representation back into crop-space boxes correctly
- [x] Project the decoded crop-space box back into frame coordinates
- [x] Verify the decoded box moves consistently with target motion
- [x] Ensure frame clipping still behaves correctly at image boundaries

Expected result after completion:

Inference uses the same representation as training, and tracker outputs should no longer collapse purely because of inconsistent decoding.

Step 5 notes:

- `SiamAPNBackend.track()` now converts normalized center-based predictions back into crop-space `xywh` before projecting them into frame coordinates.
- The inference path now mirrors the training target representation instead of consuming raw pixel-space predictions.

### Step 6: Add diagnostic checks for training behavior

Checklist:

- [x] Log or inspect target statistics after the encoding change
- [x] Log or inspect early prediction statistics during training
- [x] Confirm the loss decreases from its initial value over early batches/epochs
- [x] Check that predicted boxes are not collapsing to nearly constant values

Expected result after completion:

There is direct evidence that optimization is behaving better and that the model is learning non-trivial localization behavior.

Step 6 notes:

- User training run after the normalized box change reached `loss=0.0028 reg=0.0028` by epoch 3.
- This is a major improvement over the previous raw-pixel loss plateau around `79`, confirming that the scale mismatch was a real training issue.
- Prediction-stat diagnostic on 32 sampled pairs showed normalized target mean `[0.5088, 0.5012, 0.2676, 0.2942]` and prediction mean `[0.5195, 0.4885, 0.2813, 0.3014]`.
- Prediction std was `[0.0430, 0.0405, 0.0959, 0.0955]`, so the checkpoint is not fully collapsed to one constant box.
- Overall normalized MAE was about `0.0515`; this is reasonable for crop-space regression diagnostics but still did not translate to strong tracking IoU on `dataset1/Car_video_2`.

### Step 7: Re-run video tracking validation

Checklist:

- [x] Run the tracker on at least one known video
- [ ] Run qualitative video/app validation after a longer full-port checkpoint is trained
- [ ] Compare the longer full-port checkpoint against the previous weak baseline qualitatively
- [x] Record remaining failure modes if any still exist

Expected result after completion:

The tracker should visibly respond to target motion. If major failures remain, they are more likely due to architecture limitations rather than the original regression mismatch.

Step 7 notes:

- Added `eval-train` and `eval-public` commands for model evaluation.
- Smoke-tested `eval-train` with `uv run eval-train --limit 1 --widths 384 --override tracking.backend=siamapn --override tracking.checkpoint=checkpoints/best.pth`.
- Result on `dataset1/Car_video_2`: `avg_ms=32.53`, `mean_iou=0.1547`, `success@0.5=0.0000`, `frames=677`.
- This confirms the evaluation path works, but tracking quality is still weak on the tested sequence.
- Oracle-center diagnostic on the same sequence reached `mean_iou=0.3686` and `success@0.5=0.1403`, which means the model can localize somewhat when the search crop is centered correctly.
- This points to tracker feedback/update instability as a major remaining failure mode: once the live tracker makes a bad center update, later crops are no longer centered where the model expects.
- Added conservative center/size smoothing in `SiamAPNBackend` so one prediction cannot jump the box as aggressively.
- After smoothing, the same one-sequence benchmark was `mean_iou=0.1621`, `success@0.5=0.0015`; this is only a small improvement, so architecture/head robustness is still the main unresolved issue.

### Step 8: Evaluate whether the simplified tracker is sufficient

Checklist:

- [x] Review tracking quality after the regression fix
- [x] Decide whether the single-box pooled regressor is adequate
- [x] If not adequate, identify which SiamAPN++ mechanisms need to be restored first
- [x] Prioritize minimal additions with the highest expected impact

Expected result after completion:

There is a clear decision on whether to keep the simplified architecture or move closer to the original SiamAPN++ prediction design.

Step 8 notes:

- The single pooled 4-value regressor is not sufficient. It trains on the normalized target, but live tracking remains unstable and produces weak IoU.
- The highest-impact minimal addition is an explicit spatial localization mechanism from the correlation map, before attempting a full SiamAPN++ port.
- Decision: replace the fully pooled box regressor with a spatial center head plus a pooled size head.

### Step 9: If needed, restore higher-impact SiamAPN++ mechanisms incrementally

Checklist:

- [x] Consider adding a confidence/classification branch
- [x] Consider replacing single-box regression with structured localization outputs
- [x] Consider adding score-guided candidate selection during inference
- [x] Validate each added mechanism separately instead of changing everything at once

Expected result after completion:

Any architecture changes are introduced in a controlled way, and their effect on tracking quality can be measured step by step.

Step 9 notes:

- `models/siamapn.py` now predicts center location from a dense `center_head` over the fused correlation map instead of deriving all four box values from global average pooling.
- Width/height are still predicted by a pooled `size_head`, bounded to normalized `[0, 1]` values.
- `models/losses.py` now includes an optional center-map cross-entropy loss when `center_logits` are available, weighted at `0.1` in addition to Smooth L1 box regression.
- `train/run.py` now passes `center_logits` into the loss and reports `cls` loss separately from `reg` loss.
- `app/tracking.py` now uses `model.predict_from_features(...)`, so app inference follows the same prediction path as training.
- Existing checkpoints from the previous regressor architecture are not compatible with this new head. Retrain before running `eval-train` or app tracking with `siamapn`.
- Original score-guided candidate selection is now restored in the active SiamAPN++ backend through score fusion, penalties, a Hanning window, and best-candidate selection.

Full-port correction:

- This incremental center-head step is not enough if the goal is the real SiamAPN++ implementation.
- Replace it with the original `APN` + `clsandloc` design from `external/SiamAPN/SiamAPN++/pysot/models/utile_adapn.py`.
- Restore the original three score branches: `cls1`, `cls2`, and `cls3`.
- Restore dynamic anchor/shape prediction via `xff`/`ranchors`.
- Restore `loc` delta prediction and original score fusion/penalty/window decode from `external/SiamAPN/SiamAPN++/pysot/tracker/adsiamapn_tracker.py`.
- Restore original target tensors: `bbox`, `label_cls2`, `labelxff`, `labelcls3`, and `weightxff`.

### Step 9A: Full Original SiamAPN++ Port

Checklist:

- [x] Map original model outputs and training tensors
- [x] Confirm original MobileOne wrapper feature contract: `[384, 8/28, 8/28]` and `[256, 6/26, 6/26]`
- [x] Replace active backbone wrapper with the original SiamAPN++ MobileOne feature contract
- [x] Replace active model head with original `APN` and `clsandloc`
- [x] Replace active loss with original `cls_loss`, `loc_loss`, and `shapeloss`
- [x] Replace dataset target generation with original `AnchorTarget`/`AnchorTarget3` tensors
- [x] Replace app/evaluation decode with original score fusion, penalty, window, and LR update
- [x] Update tests for the original output contract
- [ ] Run a longer full retraining job from scratch and evaluate the resulting checkpoint

Expected result after completion:

The active codebase implements the original SiamAPN++ training and tracking contract, with MobileOne-S2 used through the same feature interface expected by the original APN heads.

Step 9A notes:

- Original model training returns and optimizes `total_loss`, `cls_loss`, `loc_loss`, and `shapeloss`.
- Original model inference returns `cls1`, `cls2`, `cls3`, and `loc`, while storing `ranchors` for tracker-side anchor generation.
- Original tracker does not directly trust one box output; it fuses scores, applies scale/aspect penalties, applies a Hanning window, chooses the best candidate, then updates center/size with a learning rate.

### Step 9B: Active Port Execution Plan

Checklist:

- [x] Freeze `external/SiamAPN/SiamAPN++` as read-only reference; do not edit it during the port
- [x] Add active SiamAPN++ constants/config fields for `template_size=127`, `search_size=287`, `output_size=21`, `anchor_stride=8`
- [x] Update `config.yaml` from search size `255` to original search size `287`
- [x] Port the original MobileOne-S2 feature wrapper into active `models/backbone/` so it returns the original feature contract
- [x] Port `APN`, `clsandloc`, `selfpointbranch`, `selfchannelbranch`, and `adcat` into active `models/`
- [x] Port the original loss functions into active `models/losses.py` or a dedicated active loss module
- [x] Port `AnchorTarget` and `AnchorTarget3` target builders into active `data/`
- [x] Replace `CompetitionSiameseDataset` output with original-style tensors: `bbox`, `label_cls2`, `labelxff`, `labelcls3`, `weightxff`
- [x] Replace `SiamAPNppMobileOne.forward(...)` with original-style training output: `total_loss`, `cls_loss`, `loc_loss`, `shapeloss`
- [x] Add original-style `template(...)` and `track(...)` model methods for inference
- [x] Replace `SiamAPNBackend` decode/update with original tracker logic: dynamic anchors, score fusion, penalties, Hanning window, LR update
- [x] Update tests to assert original output shapes and target contracts
- [ ] Run longer full retraining from scratch because old checkpoints are incompatible
- [ ] Run `eval-train` after longer retraining and compare against the current weak baseline

Expected result after completion:

The active codebase uses a faithful SiamAPN++ training and inference pipeline adapted to this repo's config, app, and evaluation commands, while preserving `external/` as the reference implementation.

Implementation notes:

- Do not keep the normalized `search_bbox_cxcywh` path as the final training contract.
- Do not keep the partial center-head architecture as the final model head.
- Avoid hardcoded `.cuda()` in the active port; all tensors/modules should follow the configured/runtime device.
- Preserve the original SiamAPN++ logic unless an adaptation is required for device handling, config integration, or dataset path integration.

Step 9B notes:

- Active geometry is now `template_size=127`, `search_size=287`, `output_size=21`, `anchor_stride=8` in `config.yaml`/`config.py`.
- Active MobileOne-S2 now returns the original SiamAPN++ feature contract: `[B, 384, 8/28, 8/28]` and `[B, 256, 6/26, 6/26]`.
- Active model now uses ported `APN` and `ClsAndLoc` modules and exposes original-style `template(...)`, `track(...)`, and training `forward(...)` outputs.
- Active dataset now emits `bbox`, `label_cls2`, `labelxff`, `labelcls3`, and `weightxff` instead of normalized `search_bbox_cxcywh`.
- Active training now optimizes `total_loss`, `cls_loss`, `loc_loss`, and `shapeloss` from the model output.
- Active SiamAPN backend now uses dynamic anchors, score fusion, penalties, Hanning window, and LR center/size update.
- Verification completed: `uv run pytest tests/test_architecture.py tests/test_tracking.py tests/test_crop_utils.py tests/test_shapes.py -q` passed with `12 passed`.
- Training smoke test completed: `uv run train --override train.epochs=1 --override train.train_samples_per_epoch=1 --override train.batch_size=1 --override train.num_workers=0 --override train.checkpoint_dir=checkpoints/smoke_full_port` completed with finite loss.

Post-audit fixes:

- Dataset search crop geometry now matches inference geometry by using `search_size / template_size` instead of a hardcoded `2.0` area scale.
- `tracking.search_crop_scale` is now documented as not controlling SiamAPN++ crop geometry; SiamAPN++ uses `model.search_size / model.template_size` to match the original training/inference contract.
- `CompetitionSiameseDataset` now accepts `output_size` and no longer hardcodes `21` for target generation.
- `train/run.py` now threads `config.model.output_size` into the dataset.
- `AnchorTarget3` now matches the original split between stricter classification positives and broader regression positives.
- `MobileOneS2Backbone.reparameterize()` now assigns the returned fused stage modules instead of discarding them.
- Post-fix verification completed: `uv run pytest tests/test_architecture.py tests/test_tracking.py tests/test_crop_utils.py tests/test_shapes.py -q` passed with `12 passed`.
- Post-fix training smoke test completed: `uv run train --override train.epochs=1 --override train.train_samples_per_epoch=1 --override train.batch_size=1 --override train.num_workers=0 --override train.checkpoint_dir=checkpoints/smoke_full_port_fix` completed with finite loss.
- First short full-port training run completed with `train.epochs=3`, `train.batch_size=8`, and `train.train_samples_per_epoch=512`.
- Training loss improved from epoch 1 to epoch 3: epoch 1 `loss=6.7993 cls=0.7407 reg=1.2662`, epoch 2 `loss=3.0767 cls=0.7611 reg=0.1582`, epoch 3 `loss=2.2652 cls=0.6159 reg=0.1125`.
- This indicates the full original-style loss is trainable and the regression branch is learning.
- Evaluation on `dataset1/Car_video_2` after the short run was still weak: `mean_iou=0.0447`, `success@0.5=0.0015`, `avg_ms=39.42`.
- The eval result is not acceptable yet; either the full head needs substantially more training, or a remaining train/inference/crop/decode mismatch still exists.
- Decord repeatedly reported corrupt/unreadable videos such as `dataset4/bike1/bike1_30.mp4` and `dataset4/bird1/bird1_30.mp4`; these are skipped/resampled but should be tracked if they slow or pollute training logs.

Bad-video handling notes:

- `CompetitionSiameseDataset` now loads known-bad video paths from `data/raw/metadata/bad_videos.txt`.
- Known bad videos are skipped before Decord tries to open them, preventing repeated Decord error logs for those files.
- Newly discovered bad videos are marked in-memory per dataset worker and skipped for the rest of that worker's lifetime.
- If a new Decord error appears once during training, add that relative video path to `data/raw/metadata/bad_videos.txt` to skip it before open attempts in future runs.

Evaluation width notes:

- `eval-train --widths` changes `tracking.track_max_width`.
- This matters for backends like CSRT that resize full frames before tracking.
- The active SiamAPN++ backend currently crops around the target and resizes to fixed `model.search_size=287`, so `--widths` is not expected to change SiamAPN++ model input size or quality.
- For normal SiamAPN++ work, omit `--widths`; use it only for backend comparison or if explicit SiamAPN frame scaling is added later.

### Step 10: Final verification and cleanup

Checklist:

- [x] Run relevant tests for the full-port code path
- [x] Confirm short full-port training runs without unstable loss behavior
- [ ] Confirm longer full-port training improves validation tracking behavior
- [ ] Confirm inference runs without point-sticking/random drift on validation videos after longer training
- [x] Update notes/docs to reflect the active full-port design

Expected result after completion:

The implementation is internally consistent, documented, and verified against both training and tracking behavior.
