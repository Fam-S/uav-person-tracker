def read_sequence_frames(sequence):
    """Yield `(frame_index, frame)` in order for one competition sequence."""

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
