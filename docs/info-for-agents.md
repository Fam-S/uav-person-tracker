# Info For Agents

The dataset merges multiple established aerial tracking benchmarks with fresh custom-annotated UAV footage. Every sequence contains a single target observed from realistic UAV trajectories that include translation, rotation, and altitude variations.

## Competition Task
- Task type: efficient aerial single-object tracking.
- One target is tracked per sequence.
- The tracker is initialized from the ground-truth bounding box in the first frame.
- Tracking must be online-only: no future frame access.
- No re-initialization is allowed unless organizers explicitly permit it.
- The practical goal is not just accuracy; the solution must also satisfy deployment-oriented efficiency constraints for UAV use.

## What Each Sequence Provides
- Native-frame-rate video (`.mp4`).
- Frame-by-frame bounding box annotations (train split only).
- One target identity per clip.
- Challenging phenomena: occlusions, rapid motion, scale changes, background clutter, and low-resolution targets.

## Dataset Split
Teams must keep the provided configuration intact.

| Split | Sequences | Duration | Purpose |
| --- | --- | --- | --- |
| `train` | 255 | ~98 min (5883 s) | Model training |
| `public_lb` | 89 | ~34 min (2028 s) | Kaggle leaderboard scoring |
| `hidden` | – | – | Final private evaluation |

## Folder Layout
```
<dataset>/<sequence_name>/
    <sequence_name>.mp4     # video at native FPS
    annotation.txt          # one bounding box per line (train only)
```

## Annotation Format
Each `annotation.txt` line maps to a frame:

```
x,y,w,h
```

`(x, y)` is the top-left corner; `w` and `h` are width and height in pixels. Frames where the target disappears use `0,0,0,0`.


## Metadata Files
Two helper files are distributed alongside the sequences.

### `contestant_manifest.json`
JSON organized by split. Every entry describes dataset source, frame count, FPS, and file paths. Example:

```json
{
  "train": {
    "dataset5/bike1": {
      "dataset":         "dataset5",
      "seq_name":        "bike1",
      "n_frames":        3085,
      "native_fps":      30,
      "video_path":      "dataset5/bike1/bike1.mp4",
      "annotation_path": "dataset5/bike1/annotation.txt"
    }
  },
  "public_lb": {
    "dataset2/basketball_player1": {
      "dataset":         "dataset2",
      "seq_name":        "basketball_player1",
      "n_frames":        450,
      "native_fps":      30,
      "video_path":      "dataset2/basketball_player1/basketball_player1.mp4",
      "annotation_path": null
    }
  }
}
```

### `sample_submission.csv`
Template with one row per frame for every public leaderboard clip:

```
id,x,y,w,h
dataset2/basketball_player1_0,0,0,0,0
dataset2/basketball_player1_1,0,0,0,0
...
```

`id` equals `<seq_id>_<frame_index>` with zero-based frame indices. Replace the zeros with predicted bounding boxes when submitting.

## Extracted Package Notes
- The extracted archive root is organized as `MTC-AIC4-data/`.
- Under that root, the data is grouped into `dataset1` through `dataset5` plus a `metadata/` directory.
- The `metadata/` directory contains `contestant_manifest.json` and `sample_submission.csv`.
- The train and public leaderboard splits listed in the docs match the extracted package summary: `train` has 255 sequences and `public_lb` has 89 sequences.

### Observed Archive Layout
```text
MTC-AIC4-data/
├── dataset[1-5]/
│   ├── [Sequence_Name]/
│   │   ├── [Sequence_Name].mp4
│   │   └── annotation.txt        # train split only
└── metadata/
    ├── contestant_manifest.json
    └── sample_submission.csv
```

### Observed Dataset Group Contents
- `dataset1`: 20 sequences, including names such as `basketball`, `basketball_2`, `Car_video`, `horse_4`, `motorcycle`, `person_3`, `surfer`, and `volleyball`.
- `dataset2`: 70 sequences, including names such as `Animal1-4`, `BMX2-5`, `Basketball`, `Car2`, `Car6`, `Car8`, `Girl1-2`, `Gull1-2`, `Horse2`, and `Kiting`.
- `dataset3`: 112 sequences, including names such as `air_conditioning_box`, `basketball_player1-4`, `bike1-9`, `building1`, and `car1-18`.
- `dataset4`: 20 sequences, including names such as `bike1`, `bird1`, `car1`, `car3`, `car6`, `car8`, `car9`, `car16`, `group1-3`, `person2`, `person4`, `person5`, `person7`, `person14`, `person17`, `person19`, `person20`, and `uav1`.
- `dataset5`: 122 sequences, including names such as `bike1-3`, `bird1`, `boat1-9`, `building1-5`, `car1-18`, `group1-3`, `person1-23`, `truck1-4`, `uav1-8`, and `wakeboard1-10`.

## PDF-Based Source Clues
- The competition PDF explicitly names four public source datasets: `UAV123`, `UAV20L`, `DTB70`, and `UAVTrack112`.
- Based on sequence counts and sequence naming patterns, the extracted package appears to align with those sources as follows.

| Package Group | Observed Count | Likely Source | Basis |
| --- | --- | --- | --- |
| `dataset2` | 70 | `DTB70` | Exact count match; sequence names like `Animal`, `BMX`, `Girl`, `Gull`, and `Kiting` fit DTB70-style naming. |
| `dataset3` | 112 | `UAVTrack112` | Exact count match with the PDF and matching UAV tracking sequence naming. |
| `dataset4` | 20 | `UAV20L` | Exact count match; sequence names such as `bike1`, `bird1`, `group1-3`, `person*`, and `uav1` look like the long-term subset derived from UAV123. |
| `dataset5` | 122 | `UAV123` | Near-exact count match to `UAV123` and strong naming overlap with classic UAV123 sequences such as `boat*`, `car*`, `person*`, `truck*`, `uav*`, and `wakeboard*`. |
| `dataset1` | 20 | Unclear | Does not cleanly match the PDF counts for the four named public datasets; may be the custom-annotated UAV footage mentioned in the docs or another organizer-curated subset. |

- This mapping is an inference from the PDF plus the extracted archive, not an explicit organizer-provided mapping.
- the competition uses several publicly available aerial SOT datasets, while the local docs also mention fresh custom-annotated UAV footage. That combination makes it plausible that one package group is organizer-created or organizer-curated rather than directly named in the PDF.

## Evaluation And Scoring

### Accuracy Metrics
- `AUC`: IoU-based success rate.
- `NormPrecision`: scale-invariant center error.
- `Precision` and `Robustness` are also defined by the organizers, but the live Kaggle leaderboard uses the blended accuracy score below.

### Accuracy Score
```text
AccuracyScore = 0.6 * AUC + 0.4 * NormPrecision
```

### Efficiency Metrics
- FLOPs
- Parameter count
- Inference latency
- Model size

- FLOPs: `0.25`
- Params: `0.15`
- Latency: `0.35`
- Model size: `0.25`

### Final Score
```text
FinalScore = (1 - 0.2) * AccuracyScore + 0.2 * EfficiencyScore
```

### Current Efficiency Budgets
The markdown docs supersede the older PDF values.

| Metric | Maximum Allowed Value |
| --- | --- |
| FLOPs | `30 GFLOPs` |
| Parameters | `50 million` |
| Inference latency | `30 ms` |
| Model size | `0.5 GB` |

### Leaderboard Clarifications
- Kaggle submissions are CSV files with predicted bounding boxes.
- The public Kaggle leaderboard reflects `AccuracyScore` only.
- `EfficiencyScore` is measured locally by the organizers after Phase I for qualifying teams.
- Advancement and final ranking use the full score, not just the public leaderboard.
- Final evaluation also includes a private test set.

## Model Usage Rules
- Pretrained models are allowed.
- Fine-tuning is allowed.
- The final solution must still satisfy the competition constraints.
- The organizer Q&A says the solution must include meaningful participant contribution and cannot just be an off-the-shelf pretrained model.

## Additional Notes For Agents
- In plain terms, this is competition-provided UAV single-object tracking data: video clips, bounding-box annotations for training, and metadata for evaluation and submission generation.
- The docs describe the data as a merge of existing aerial tracking benchmarks plus custom UAV footage, but they do not explicitly name the original benchmark datasets.
- The package examples reference internal source labels such as `dataset1`, `dataset2`, and `dataset5` rather than public benchmark names.
- Prefer the markdown docs in this directory over the PDF for current constraint values, scoring, and operational rules.

## License & Compliance
- Usage remains subject to official competition rules and any upstream dataset licenses.
- Keep the prescribed split boundaries and refrain from manual relabeling unless instructions change.
- Competition data is for MTC-AIC4 participation and Kaggle discussion forums only.
- Do not transmit, duplicate, or share the competition data outside the competition scope.
- Do not redistribute dataset contents as part of submissions.
