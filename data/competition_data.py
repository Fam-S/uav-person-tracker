from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


VALID_SPLITS = {"train", "public_lb"}
MANIFEST_PATH = Path("metadata") / "contestant_manifest.json"
CLEAN_MANIFEST_PATH = Path("metadata") / "contestant_manifest.cleaned.json"


class SequenceRecord:
    """One sequence resolved from the competition manifest.

    Train sequences carry full frame-by-frame boxes.
    Public leaderboard sequences carry only the first-frame init box.
    """

    def __init__(
        self,
        seq_id,
        split,
        dataset,
        seq_name,
        video_path,
        n_frames,
        native_fps,
        init_box_xywh,
        gt_boxes_xywh,
    ):
        self.seq_id = seq_id
        self.split = split
        self.dataset = dataset
        self.seq_name = seq_name
        self.video_path = video_path
        self.n_frames = n_frames
        self.native_fps = native_fps
        self.init_box_xywh = init_box_xywh
        self.gt_boxes_xywh = gt_boxes_xywh

    def __repr__(self):
        return (
            f"SequenceRecord(seq_id={self.seq_id!r}, split={self.split!r}, "
            f"n_frames={self.n_frames!r})"
        )


def load_sequences(raw_root, split, manifest_path=None):
    """Load one official split from the competition manifest.
    raw_root (raw data directory) should point to the directory containing the `metadata` folder and all video/annotation files.
    split (str): The split to load.
    """

    split = _normalize_split(split)
    raw_root = Path(raw_root)

    # The manifest already defines the official split membership and file paths,
    # so we use it as the single source of truth instead of scanning directories.
    manifest = _load_manifest(raw_root, manifest_path=manifest_path)
    entries = manifest[split]

    sequences = []
    for seq_id, entry in entries.items():
        annotation_path = raw_root / entry["annotation_path"]
        boxes = _load_annotation_boxes(annotation_path)

        n_frames = int(entry["n_frames"])
        if split == "train":
            # Train annotations contain one box per frame, so they should line up
            # exactly with the frame count reported by the manifest.
            if len(boxes) != n_frames:
                raise ValueError(
                    f"{seq_id} has {len(boxes)} annotation rows, expected {n_frames}."
                )
            gt_boxes_xywh = boxes
        else:
            # Public leaderboard clips only expose the init box for frame 0.
            if len(boxes) != 1:
                raise ValueError(
                    f"{seq_id} should have exactly 1 init box, found {len(boxes)}."
                )
            gt_boxes_xywh = None

        sequences.append(
            SequenceRecord(
                seq_id=seq_id,
                split=split,
                dataset=entry["dataset"],
                seq_name=entry["seq_name"],
                video_path=raw_root / entry["video_path"],
                n_frames=n_frames,
                native_fps=int(entry["native_fps"]),
                init_box_xywh=boxes[0].copy(),
                gt_boxes_xywh=gt_boxes_xywh,
            )
        )

    return sequences


def build_clean_train_manifest(raw_root, output_path=None):
    """Write a manifest with unreadable train videos removed and return its path."""

    raw_root = Path(raw_root)
    output_path = Path(output_path) if output_path is not None else raw_root / CLEAN_MANIFEST_PATH
    manifest = _load_manifest(raw_root)

    cleaned_train = {}
    skipped_seq_ids = []
    for seq_id, entry in manifest["train"].items():
        if _is_video_readable(raw_root / entry["video_path"]):
            cleaned_train[seq_id] = entry
        else:
            skipped_seq_ids.append(seq_id)

    cleaned_manifest = dict(manifest)
    cleaned_manifest["train"] = cleaned_train

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned_manifest, handle, indent=2)

    return output_path, skipped_seq_ids


def _normalize_split(split):
    """Validate that the requested split is supported."""

    if split not in VALID_SPLITS:
        valid = ", ".join(sorted(VALID_SPLITS))
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {valid}.")
    return split


def _load_manifest(raw_root, manifest_path=None):
    """Load the competition manifest from the raw data root."""

    manifest_path = Path(manifest_path) if manifest_path is not None else raw_root / MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    missing = VALID_SPLITS.difference(manifest)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing split entries: {names}.")

    return manifest


def _is_video_readable(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False
    try:
        ok, frame = capture.read()
        return bool(ok and frame is not None)
    finally:
        capture.release()


def _load_annotation_boxes(annotation_path):
    """Parse an annotation file into an ``(N, 4)`` float32 ``xywh`` array."""

    rows = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                # Ignore empty lines.
                continue

            # The package is mostly comma-separated, but some absent-target rows are
            # written as `0 0 0 0`, so normalizing both forms keeps the parser small.
            parts = line.replace(",", " ").split()
            if len(parts) != 4:
                raise ValueError(
                    f"{annotation_path} line {line_number} should have 4 values, found {len(parts)}."
                )
            # Parse one box as [x, y, w, h].
            rows.append([float(part) for part in parts])

    if not rows:
        raise ValueError(f"{annotation_path} does not contain any boxes.")

    # Keep numeric type consistent across the pipeline.
    return np.asarray(rows, dtype=np.float32)
