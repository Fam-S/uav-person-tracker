from __future__ import annotations

import os


def read_sequence_frames(sequence):
    """Yield `(frame_index, frame_bgr)` in order for one competition sequence.

    Evaluation defaults to OpenCV's streaming reader because public evaluation is
    strictly sequential and Decord 0.6 has known memory-growth reports in some
    environments. Set `UAV_EVAL_VIDEO_READER=decord` to opt into Decord.
    """

    if os.getenv("UAV_EVAL_VIDEO_READER", "cv2").lower() == "decord":
        try:
            yield from _read_sequence_frames_decord(sequence)
            return
        except Exception:
            pass
    yield from _read_sequence_frames_cv2(sequence)


def _read_sequence_frames_decord(sequence):
    import gc

    import numpy as np
    from decord import VideoReader, cpu
    from decord import logging as decord_logging

    decord_logging.set_level(decord_logging.QUIET)
    reader = VideoReader(str(sequence.video_path), ctx=cpu(0))
    if len(reader) < sequence.n_frames:
        raise RuntimeError(
            f"Video ended early for {sequence.seq_id}: {len(reader)} frames, expected {sequence.n_frames}."
        )

    try:
        for frame_index in range(sequence.n_frames):
            frame = reader.next()
            frame_rgb = frame.asnumpy()
            frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
            del frame, frame_rgb
            yield frame_index, frame_bgr
    finally:
        reader = None
        gc.collect()


def _read_sequence_frames_cv2(sequence):
    import cv2

    capture = cv2.VideoCapture(str(sequence.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {sequence.video_path}")

    try:
        for frame_index in range(sequence.n_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Video ended early for {sequence.seq_id} at frame {frame_index}."
                )
            yield frame_index, frame
    finally:
        capture.release()
