import argparse
from pathlib import Path
from time import perf_counter

from config import load_config
from app.tracking import create_backend
from data import load_sequences, read_sequence_frames, write_submission_csv
from tqdm import tqdm


def run_public_lb(sequences, make_backend):
    """Run one backend over the public leaderboard split and collect rows by id."""

    # predictions will map each row id (e.g. "dataset1/Car_video_0_5") to a bbox tuple (x, y, w, h).
    predictions = {}
    total_latency_ms = 0.0
    total_tracked_frames = 0

    progress = tqdm(sequences, desc="public_lb", unit="seq")
    for sequence in progress:
        # Run the tracker on every frame of this sequence and get back its predictions + timing.
        sequence_predictions, latency_ms, tracked_frames = run_sequence(sequence, make_backend)

        # Merge this sequence's predictions into the overall dict.
        predictions.update(sequence_predictions)
        total_latency_ms += latency_ms
        total_tracked_frames += tracked_frames

        # Guard against division by zero if a sequence had no tracked frames.
        avg_ms = latency_ms / tracked_frames if tracked_frames else 0.0
        progress.set_postfix(seq=sequence.seq_id, avg_ms=f"{avg_ms:.2f}, frames={sequence.n_frames}")

    overall_avg = total_latency_ms / total_tracked_frames if total_tracked_frames else 0.0
    print(f"overall average tracking latency: {overall_avg:.2f} ms")
    return predictions


def run_sequence(sequence, make_backend):
    # Create a fresh backend instance for every sequence so state doesn't leak between them.
    backend = make_backend()
    predictions = {}
    latency_ms = 0.0
    tracked_frames = 0

    try:
        for frame_index, frame in read_sequence_frames(sequence):
            # Build the submission row id: "<seq_id>_<frame_index>", e.g. "dataset1/Car_video_0_3".
            row_id = f"{sequence.seq_id}_{frame_index}"

            if frame_index == 0:
                # The first frame is the template frame — we give the tracker the ground-truth box
                # so it knows what the target looks like. We don't count this as a tracked frame.
                init_bbox = _clip_bbox(sequence.init_box_xywh, frame.shape)
                predictions[row_id] = init_bbox
                backend.initialize(frame, init_bbox)
                continue

            # For every frame after the first, ask the tracker where the target is now.
            result = backend.track(frame)
            predictions[row_id] = _clip_bbox(result.bbox, frame.shape)
            latency_ms += result.latency_ms  # accumulate how long tracking took
            tracked_frames += 1
    finally:
        # Always release backend resources (GPU memory, file handles, etc.) even if an error occurs.
        backend.reset()

    return predictions, latency_ms, tracked_frames


def _clip_bbox(bbox, frame_shape):
    """Clamp a bounding box so it stays fully inside the frame, returning (0,0,0,0) if invalid."""

    # If the tracker returned nothing, output a zero box (competition expects this for missing detections).
    if bbox is None:
        return (0, 0, 0, 0)

    # frame_shape is (height, width, channels) — numpy image convention.
    frame_h, frame_w = frame_shape[:2]

    # Round floats to the nearest pixel and convert to int.
    x, y, w, h = [int(round(value)) for value in bbox]

    # Clamp the top-left corner so it doesn't go outside the frame boundaries.
    # After this, x is guaranteed to be in [0, frame_w - 1] and y in [0, frame_h - 1].
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))

    # Width and height can't be negative.
    w = max(0, w)
    h = max(0, h)

    # A box with zero area is useless — treat it as a missed detection.
    if w == 0 or h == 0:
        return (0, 0, 0, 0)

    # Clamp width/height so the box doesn't extend beyond the right/bottom edge.
    # Because x <= frame_w - 1, (frame_w - x) >= 1, so w stays positive after this.
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    return (x, y, w, h)


def main():
    parser = argparse.ArgumentParser(description="Run a backend over public_lb and write a submission CSV.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="evaluation/submissions/public_lb_submission.csv")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Override config value, e.g. --override tracking.backend=csrt")
    args = parser.parse_args()

    overrides = {}
    for item in args.override:
        key, _, value = item.partition("=")
        if not key or not value:
            parser.error(f"Invalid --override format: '{item}'. Use KEY=VALUE.")
        overrides[key] = value

    project_config = load_config(args.config, overrides=overrides or None)

    # Load only the sequences that belong to the public leaderboard split.
    sequences = load_sequences(args.raw_root, "public_lb")

    # make_backend is a factory: calling it returns a fresh, ready-to-use tracker instance.
    def make_backend():
        return create_backend(project_config.tracking)

    # Time the full evaluation from start to finish.
    start = perf_counter()
    predictions = run_public_lb(sequences, make_backend)
    write_submission_csv(args.raw_root, args.output, predictions)
    elapsed = perf_counter() - start

    print(f"wrote submission to {Path(args.output)}")
    print(f"total runtime: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
