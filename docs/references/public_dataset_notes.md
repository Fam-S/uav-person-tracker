## Public Dataset Notes

This note summarizes how the available competition dataset compares to the original goal of this project, which is **UAV person tracking**.

---

## Is the available dataset good enough?

Yes, the available competition dataset is **good for the competition-style task**, but it is **not perfectly matched** to a person-only tracker.

Why:

- the competition task is **class-agnostic aerial single-object tracking**
- the tracker is initialized from the first-frame ground-truth box
- tracking is online-only
- no future frame access is allowed

The current data is strong for aerial tracking because it already includes:

- occlusions
- rapid motion
- scale changes
- background clutter
- low-resolution targets

However, it is not person-only. The available sequences include many categories such as:

- bikes
- birds
- boats
- cars
- groups
- persons
- UAVs
- wakeboards

So the conclusion is:

- **for general aerial single-object tracking:** the current dataset is strong
- **for UAV person tracking specifically:** it is useful, but category-mismatched

---

## Recommendation

Use the current competition dataset as the **main training and evaluation source** for aerial SOT.

If the real target is specifically **tracking people from UAV footage**, then it is a good idea to **supplement** it with more human-focused aerial datasets.

Recommended supplementary direction:

- **VisDrone**
- **Okutama-Action**
- **UAVHuman** if available and allowed for the intended use

---

## VisDrone

Repository:

- https://github.com/VisDrone/VisDrone-Dataset

VisDrone is a strong public UAV dataset family because it focuses on drone imagery and includes several challenge tracks relevant to edge vision and tracking.

The challenge mainly focuses on five tasks:

1. **Task 1: object detection in images**
   - Detect predefined object categories such as cars and pedestrians from individual drone images.

2. **Task 2: object detection in videos**
   - Similar to Task 1, but detection is performed on video frames.

3. **Task 3: single-object tracking**
   - Estimate the state of a target, indicated in the first frame, across subsequent video frames.

4. **Task 4: multi-object tracking**
   - Recover the trajectories of multiple objects in each video frame.

5. **Task 5: crowd counting**
   - Count persons in each video frame.

---

## Why VisDrone is Useful Here

VisDrone is especially useful because it adds more UAV-specific visual conditions such as:

- small targets
- cluttered backgrounds
- frequent occlusion
- aerial viewpoint changes
- dense human and vehicle scenes

For this project, VisDrone is most useful as:

- a **supplementary dataset** for person-focused aerial scenes
- a source of more difficult UAV visual conditions
- a reference for both **single-object tracking** and broader UAV perception tasks

---

## Practical Takeaway

- Keep the current competition dataset as the **main dataset** for the original competition-like tracking task.
- Add **VisDrone** if you want stronger alignment with **real UAV person tracking conditions**.
- If needed, extract person-centered tracklets from broader aerial datasets to make training more person-specific.

---

## Ranked Public Datasets for UAV Person Tracking

This ranking is for the specific goal of **UAV person tracking**, not just generic aerial single-object tracking.

### 1. VisDrone

**Best overall first choice**

Why:

- real UAV imagery
- strong pedestrian / person presence
- includes single-object tracking, multi-object tracking, video detection, and crowd counting
- very useful for small targets, clutter, occlusion, and realistic drone viewpoints

Best use:

- strongest all-around public dataset for UAV person tracking support

### 2. UAVHuman

**Best person-specific UAV dataset**

Why:

- more human-centered than general aerial SOT datasets
- useful for person appearance variation and human-focused UAV scenes
- better semantic fit when the real target is specifically people

Best use:

- best choice when making the model more human-specialized

### 3. Okutama-Action

**Best for aerial human motion and interactions**

Why:

- focused on humans from UAV views
- useful for occlusion, pose variation, group interaction, and outdoor human motion
- not a pure SOT benchmark, but very useful for extracting person tracklets

Best use:

- robustness training for difficult human motion and scene changes

### 4. UAV123 / UAV20L

**Best classic UAV SOT benchmarks**

Why:

- important benchmarks for online single-object tracking
- **UAV20L** is especially useful for long-term tracking
- not person-only, so they are better as tracking benchmarks than as person-specialized datasets

Best use:

- tracker-style evaluation and long-term behavior testing

### 5. DTB70 / UAVTrack112

**Useful supplementary aerial SOT datasets**

Why:

- helpful for aerial tracking diversity
- less specifically aligned with person-focused tracking

Best use:

- additional SOT diversity after stronger person/UAV datasets are already included

## Recommended Dataset Strategy

- If only one public dataset is added: **VisDrone**
- If two are added: **VisDrone + UAVHuman**
- If long-term tracking evaluation matters most: include **UAV20L**
- For person-focused training, it is often helpful to extract **single-target person tracklets** from broader detection or MOT datasets
