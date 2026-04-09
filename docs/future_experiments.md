## Future Experiments

This note captures a future direction for handling **long-term tracking** and **occlusions** under **CPU-only** constraints.

---

## Goal

Improve robustness when:

- the target is partially occluded
- the target is fully occluded
- the target leaves the local search area
- the target re-enters the scene later
- the tracker starts drifting after a weak prediction

---

## Main Idea

Do not rely only on a stronger backbone.

Instead, build a **system around the tracker**:

- fast local Siamese tracker for normal frames
- confidence-based failure detection
- template memory for appearance changes
- motion prior for search placement
- fallback re-detection when the target is lost

This is more suitable for long-term tracking than only increasing backbone size.

---

## Proposed Architecture

### 1. Short-term local tracker

- Use the main Siamese tracker every frame
- Track inside a cropped search region around the last known target location
- Keep this path lightweight because it runs continuously

### 2. Confidence / loss estimator

Use tracking signals to estimate whether the tracker is reliable.

Possible signals:

- response-map peak value
- peak-to-sidelobe ratio
- classification confidence
- sudden box size or position instability

This module decides whether the system stays in normal tracking mode or switches to recovery behavior.

### 3. Template memory bank

Keep a small set of templates instead of only one:

- the initial template
- a few recent high-confidence templates

Benefits:

- better robustness to appearance change
- less risk than replacing the template every frame

Important rule:

- **do not update templates during uncertain or occluded states**

### 4. Motion prior

Use a lightweight motion model such as:

- Kalman filter
- constant-velocity prediction

Purpose:

- predict the next likely target position
- place the next search crop more intelligently
- reduce search cost during temporary uncertainty

### 5. Recovery / re-detection module

If confidence drops too much:

- first expand the search region gradually
- if still lost, switch to low-frequency global re-detection

This re-detection stage can use:

- a lightweight detector
- or a wider global search strategy

Important idea:

- re-detection should **not** run every frame on CPU
- it should run only when the local tracker is unreliable

### 6. Tracking state machine

Use three states:

#### Tracking
- normal local tracking
- small search region
- template updates allowed only when confidence is high

#### Uncertain
- confidence is falling
- expand search region moderately
- freeze template update
- rely more on motion prior

#### Lost / Re-detecting
- tracker confidence is too low for too long
- no template update
- run re-detection occasionally until the target is found again

---

## Why This Is Better Than Only a Stronger Backbone

Occlusion is mostly a **system-level problem**, not just a feature-extractor problem.

A stronger backbone does not automatically solve:

- full disappearance
- target re-entry
- drift after incorrect updates
- target leaving the local crop

The biggest gains usually come from:

- detecting failure early
- preventing bad template updates
- having a recovery path

---

## CPU-Only Design Rules

To stay practical on CPU:

- keep the local tracker as the default fast path
- avoid full-frame processing during normal tracking
- run re-detection only on failure
- keep template memory small
- use a lightweight motion model
- keep matching and heads simple

---

## Recommended Future Experiment

### Primary future direction

**Siamese tracker + long-term recovery state machine**

Components:

- local Siamese tracker
- confidence gating
- template memory bank
- motion prior
- fallback re-detection

This gives the best chance of improving long-term robustness without turning the whole system into a heavy per-frame pipeline.

### Alternative non-Siamese direction

**Non-Siamese online discriminative tracker**

This is the best contrasting experiment if we want something meaningfully different from the primary Siamese design.

Suggested form:

- backbone: **ShuffleNetV2-1.0x**
- online target classifier / discriminative filter
- cropped local search region
- confidence-gated online updates
- lightweight scale or box refinement
- lost handling that outputs `0,0,0,0` when the target is absent for too long

Why this is a strong secondary direction:

- it is **not Siamese-based**
- it is naturally **class-agnostic**, which fits the competition because the task is **single-object tracking per sequence**, but the dataset spans **many target categories** rather than one fixed class
- it adapts online to the specific target appearance
- it gives a more meaningful comparison than just swapping backbones inside another Siamese tracker

Main risks:

- online model updates can drift if confidence control is weak
- update frequency must stay conservative for CPU efficiency
- implementation is usually more delicate than a plain Siamese matching pipeline

This makes it a good secondary experiment when the goal is to compare:

- **fixed-template matching** vs.
- **online target-adaptive tracking**

### Classic lightweight baseline

**DCF / correlation-filter tracker**

This is a useful classic baseline if we want a very cheap and CPU-friendly non-deep alternative.

Examples of this family include:

- KCF
- CSRT
- CSR-DCF

Why it is worth testing:

- extremely CPU-friendly
- simple to benchmark
- useful as a classical reference against learned trackers
- still follows the same competition setup of **one target per sequence initialized in the first frame**

Why it still fits the competition:

- it is **class-agnostic**
- it does not assume a fixed object category such as only people or only cars
- that makes it compatible with a benchmark where each clip has one target, but the overall dataset includes many target types

Main limitations:

- weaker under heavy occlusion
- weaker under large appearance change
- weaker under re-detection and long-term recovery
- more likely to lose the target in difficult UAV scenes than stronger learned trackers

---

## Summary

For long-term tracking and occlusion handling, the best future direction is:

**fast local tracker + reliability estimation + memory + recovery mode**

That is a more CPU-friendly and realistic solution than only scaling up the backbone.

For a different architecture family from the primary tracker, the strongest future experiment is:

**a lightweight online discriminative tracker with ShuffleNetV2-1.0x**

This provides a clear non-Siamese comparison for CPU-only single-object tracking.

A useful classic benchmark alongside it is:

**a DCF / correlation-filter tracker**

This gives a cheap, class-agnostic baseline for the same competition setting.
