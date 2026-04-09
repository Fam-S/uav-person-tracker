## Future Improvements for the Primary Experiment

This document records practical ways to improve the **primary experiment** while keeping the same core architecture:

- **Symmetric Siamese tracker**
- **MobileOne-S0 backbone on both branches**
- **CPU-only target**

The goal is to improve robustness, especially for:

- occlusion
- drift
- appearance change
- temporary target loss

---

## Main Principle

Do not start by making the backbone much bigger.

For this setup, the highest-value improvements come from **system-level robustness**, not only from increasing model size.

---

## Recommended Improvements

### 1. Confidence-gated template updates

- Update the template only when tracking confidence is high
- Freeze updates when confidence drops
- Helps prevent drift during occlusion or ambiguous frames

Why it matters:

- one of the cheapest and most effective improvements
- avoids corrupting the target representation

### 2. Template memory bank

- Keep more than one template instead of only one
- Store:
  - the initial template
  - a few recent high-confidence templates

Why it matters:

- improves robustness to appearance change
- gives the tracker multiple reliable target views

### 3. Three-state tracking logic

Use a simple state machine:

- **tracking**
- **uncertain**
- **lost**

Why it matters:

- allows different behavior depending on confidence
- prevents bad updates when the tracker is unreliable
- creates a clean path for recovery behavior

### 4. Adaptive search region

- Use a small crop when confidence is high
- Expand the crop when confidence becomes weak

Why it matters:

- improves recovery chances
- keeps normal tracking fast on CPU
- avoids full-frame search in most frames

### 5. Motion prior

- Use a lightweight motion model such as:
  - Kalman filter
  - constant-velocity prediction

Why it matters:

- helps place the next search crop more intelligently
- improves stability when the response map is weak

### 6. Train for failure cases

Add training data or augmentation for:

- occlusion
- motion blur
- scale changes
- similar-person distractors
- partial out-of-frame views

Why it matters:

- improves robustness without changing inference architecture
- especially useful for UAV scenes with clutter and rapid motion

---

## Best Improvement Order

If improvements are added gradually, the best order is:

1. **Confidence-gated updates**
2. **Template memory bank**
3. **Three-state tracking logic**
4. **Adaptive search region**
5. **Motion prior**
6. **Failure-case training improvements**

---

## Recommended Next Step

The best next improvement is:

**confidence-gated updates + a small template memory bank**

Why:

- large robustness gain
- low CPU cost
- keeps the primary architecture simple and fast

---

## Summary

The best way to enhance the primary experiment is to keep the **MobileOne-S0 symmetric Siamese core** and add:

- safer update logic
- memory
- confidence-based behavior
- lightweight recovery support

That is likely to help more than simply scaling the backbone.
