import argparse

from config import load_config
from app.tracking import create_backend
from data import load_sequences, read_sequence_frames


def benchmark_widths(sequences, config, widths):
    results = []
    for width in widths:
        config.tracking.track_max_width = width

        total_latency_ms = 0.0
        total_tracked_frames = 0
        total_iou = 0.0
        total_success_50 = 0
        used_sequences = 0

        for sequence in sequences:
            try:
                latency_ms, tracked_frames, iou_sum, success_50 = run_sequence_benchmark(sequence, config)
            except RuntimeError as exc:
                print(f"skip {sequence.seq_id}: {exc}")
                continue

            total_latency_ms += latency_ms
            total_tracked_frames += tracked_frames
            total_iou += iou_sum
            total_success_50 += success_50
            used_sequences += 1

        avg_ms = total_latency_ms / total_tracked_frames if total_tracked_frames else 0.0
        mean_iou = total_iou / total_tracked_frames if total_tracked_frames else 0.0
        success_50 = total_success_50 / total_tracked_frames if total_tracked_frames else 0.0

        results.append(
            {
                "width": width,
                "avg_ms": avg_ms,
                "mean_iou": mean_iou,
                "success_50": success_50,
                "frames": total_tracked_frames,
                "sequences": used_sequences,
            }
        )

    return results


def run_sequence_benchmark(sequence, config):
    backend = create_backend(config.tracking)
    latency_ms = 0.0
    tracked_frames = 0
    iou_sum = 0.0
    success_50 = 0

    try:
        for frame_index, frame in read_sequence_frames(sequence):
            gt_bbox = _clip_bbox(sequence.gt_boxes_xywh[frame_index], frame.shape)

            if frame_index == 0:
                backend.initialize(frame, gt_bbox)
                continue

            result = backend.track(frame)
            pred_bbox = _clip_bbox(result.bbox, frame.shape)
            iou = compute_iou(pred_bbox, gt_bbox)

            latency_ms += result.latency_ms
            tracked_frames += 1
            iou_sum += iou
            success_50 += int(iou >= 0.5)
    finally:
        backend.reset()

    return latency_ms, tracked_frames, iou_sum, success_50


def pick_sequences(sequences, limit):
    return sequences[:limit]


def compute_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def _clip_bbox(bbox, frame_shape):
    if bbox is None:
        return (0, 0, 0, 0)

    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = [int(round(value)) for value in bbox]

    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(0, min(w, frame_w - x))
    h = max(0, min(h, frame_h - y))
    return (x, y, w, h)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the configured backend at multiple widths.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--widths", nargs="+", type=int, default=[640, 512, 480, 384, 320])
    args = parser.parse_args()

    config = load_config(args.config)
    train_sequences = load_sequences(args.raw_root, "train")
    picked = pick_sequences(train_sequences, args.limit)

    print("benchmark sequences:")
    for sequence in picked:
        print(f"- {sequence.seq_id} ({sequence.n_frames} frames)")

    print()
    print(f"backend: {config.tracking.backend}")
    print()
    results = benchmark_widths(picked, config, args.widths)

    print("width  avg_ms  mean_iou  success@0.5  frames  seqs")
    for row in results:
        print(
            f"{row['width']:>5}  {row['avg_ms']:>6.2f}  {row['mean_iou']:>8.4f}  "
            f"{row['success_50']:>11.4f}  {row['frames']:>6}  {row['sequences']}"
        )


if __name__ == "__main__":
    main()
