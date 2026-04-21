## Old Design Future Experiments and Improvements

This note captures a future direction for handling **long-term tracking** and **occlusions** under **CPU-only** constraints.

It refers to the previous Siamese tracker design and should be treated as old-design planning notes while the primary architecture is being changed.

---

## Base Tracker Structure

Before discussing improvements, it helps to anchor the discussion in the base lightweight Siamese tracker structure.

### Main components

1. **Template branch**
   - Takes the target crop from the first frame
   - Encodes what the object looks like

2. **Search branch**
   - Takes the search region from the current frame
   - Looks for the target near the previous location

3. **Shared lightweight backbone**
   - Extracts features for both branches using the same network
   - The exact backbone can change over time; the key idea is lightweight shared feature extraction

4. **Lightweight fusion or feature enhancement**
   - Refines or fuses features before matching when needed
   - Helps keep the tracker efficient while improving feature quality

5. **Matching module**
   - Usually some form of cross-correlation
   - Compares template features with search features

6. **Prediction head**
   - Outputs a classification score for target location
   - Outputs bounding box regression for target size and position

7. **Tracking logic and post-processing**
   - Selects the best response
   - Applies smoothing, scale update, and crop update for the next frame

### Simple flow

Template crop + search crop
-> shared lightweight feature extractor
-> feature fusion / correlation
-> score map + box prediction
-> updated target location

### In short

A lightweight Siamese tracker is basically:

- two inputs: template + search
- one shared lightweight backbone
- one lightweight matching stage
- one small prediction head

That is why this family is attractive for **CPU-only edge tracking**.

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

## Improvement Roadmap

The best path is usually:

1. start with a clean lightweight Siamese baseline
2. strengthen it with low-cost robustness improvements
3. only then test broader long-term recovery extensions or alternative tracker families

---

## Immediate Improvements for the Primary Tracker

Before testing broader architecture changes, the highest-value path is to strengthen the current Siamese tracker with low-cost system improvements.

### Recommended improvements

#### 1. Confidence-gated template updates

- Update the template only when tracking confidence is high
- Freeze updates when confidence drops
- Helps prevent drift during occlusion or ambiguous frames

Why it matters:

- one of the cheapest and most effective improvements
- avoids corrupting the target representation

#### 2. Template memory bank

- Keep more than one template instead of only one
- Store:
  - the initial template
  - a few recent high-confidence templates

Why it matters:

- improves robustness to appearance change
- gives the tracker multiple reliable target views

#### 3. Three-state tracking logic

Use a simple state machine:

- **tracking**
- **uncertain**
- **lost**

Why it matters:

- allows different behavior depending on confidence
- prevents bad updates when the tracker is unreliable
- creates a clean path for recovery behavior

#### 4. Adaptive search region

- Use a small crop when confidence is high
- Expand the crop when confidence becomes weak

Why it matters:

- improves recovery chances
- keeps normal tracking fast on CPU
- avoids full-frame search in most frames

#### 5. Motion prior

- Use a lightweight motion model such as:
  - Kalman filter
  - constant-velocity prediction

Why it matters:

- helps place the next search crop more intelligently
- improves stability when the response map is weak

#### 6. Train for failure cases

Add training data or augmentation for:

- occlusion
- motion blur
- scale changes
- similar-person distractors
- partial out-of-frame views

Why it matters:

- improves robustness without changing inference architecture
- especially useful for UAV scenes with clutter and rapid motion

### Best improvement order

If improvements are added gradually, the best order is:

1. **Confidence-gated updates**
2. **Template memory bank**
3. **Three-state tracking logic**
4. **Adaptive search region**
5. **Motion prior**
6. **Failure-case training improvements**

### Best next step

The best immediate upgrade is:

**confidence-gated updates + a small template memory bank**

Why:

- large robustness gain
- low CPU cost
- keeps the primary architecture simple and fast

---

## Long-Term Recovery Architecture

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

## Alternative Experiments

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
