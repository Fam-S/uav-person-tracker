# Evaluation

## Purpose

This directory contains simple evaluation and submission utilities.

Right now the main tool is:

- `basic_eval.py` - runs the configured tracker backend over the competition `public_lb` split and writes a Kaggle submission CSV
- `benchmark_backend.py` - benchmarks the configured backend on labeled training videos at multiple `track_max_width` values

## What It Uses

`basic_eval.py` reads from:

- `data/raw/metadata/contestant_manifest.json` - official split membership and video paths
- `data/raw/metadata/sample_submission.csv` - official submission row order and ids
- `config.yaml` - shared project settings, including tracker backend and tracking settings
- `app/app_config.yaml` - app-only window, overlay, and tracker override settings

## Current Flow

For each `public_lb` sequence, the script:

1. loads the first-frame init box from the competition metadata
2. opens the video and reads frames in order
3. initializes the configured backend on frame 0
4. tracks frames `1..N-1`
5. writes predictions into the exact Kaggle template order

Frame `0` uses the provided init box directly.

If the tracker returns no box, the script writes `0,0,0,0` for that frame.

## How To Run

Set the backend you want in `config.yaml` first.

Example:

```yaml
tracking:
  backend: csrt
```

Then run:

```bash
uv run python -m evaluation.basic_eval
```

Or with explicit paths:

```bash
uv run python -m evaluation.basic_eval --raw-root data/raw --config config.yaml --output submissions/public_lb_submission.csv
```

## Output

Default output:

```text
submissions/public_lb_submission.csv
```

The output CSV matches the id order from:

```text
data/raw/metadata/sample_submission.csv
```

That means it is ready to submit to the Kaggle competition baseline as long as the backend is valid.

## Benchmarking

If you want to compare speed and tracking quality at different `track_max_width` values, run:

```bash
uv run python -m evaluation.benchmark_backend --limit 4
```

This uses the backend already selected in `config.yaml` and reports:

- average tracking latency per frame
- mean IoU against train annotations
- success rate at IoU `0.5`

## Notes

- The script is backend-agnostic. It only depends on the backend interface used in `app/tracking.py`.
- To swap trackers later, keep the same backend interface and update `config.yaml`.
- The progress bar is sequence-level, and the script also prints a short per-sequence latency summary.
