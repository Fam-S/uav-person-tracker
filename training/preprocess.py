from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.dataset_loader import (
    ATTRIBUTE_NAMES,
    UAV123DatasetLoader,
    get_processed_dataset_paths,
    serialize_sequence_annotations,
    serialize_sequence_manifest,
)


def build_split_lists(
    sequence_records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    grouped: dict[str, list[str]] = {}
    for record in sequence_records:
        grouped.setdefault(record["base_sequence"], []).append(record["sequence_name"])

    base_sequences = sorted(grouped)
    random.Random(seed).shuffle(base_sequences)
    val_count = max(1, int(round(len(base_sequences) * val_ratio))) if base_sequences else 0
    val_bases = set(base_sequences[:val_count])

    train_names: list[str] = []
    val_names: list[str] = []
    for base_sequence in sorted(grouped):
        target = val_names if base_sequence in val_bases else train_names
        target.extend(sorted(grouped[base_sequence]))

    return train_names, val_names


def run_preprocess(
    dataset_root: str | Path,
    output_dir: str | Path,
    subset: str = "UAV123",
    config_filename: str = "configSeqs.m",
    val_ratio: float = 0.2,
    seed: int = 7,
) -> dict[str, Any]:
    loader = UAV123DatasetLoader(
        dataset_root=dataset_root,
        subset=subset,
        config_filename=config_filename,
        processed_root=None,
        split=None,
    )
    sequences = loader.load_sequences_from_raw()
    sequence_records = [serialize_sequence_manifest(sequence) for sequence in sequences]
    annotation_records = {
        sequence.name: serialize_sequence_annotations(sequence)
        for sequence in sequences
    }

    train_names, val_names = build_split_lists(sequence_records, val_ratio=val_ratio, seed=seed)
    paths = get_processed_dataset_paths(output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.train_split.parent.mkdir(parents=True, exist_ok=True)

    paths.manifest.write_text(
        json.dumps({"subset": subset, "sequences": sequence_records}, indent=2),
        encoding="utf-8",
    )
    paths.annotations_cache.write_text(
        json.dumps(annotation_records, indent=2),
        encoding="utf-8",
    )
    paths.train_split.write_text("\n".join(train_names) + "\n", encoding="utf-8")
    paths.val_split.write_text("\n".join(val_names) + "\n", encoding="utf-8")

    invalid_reason_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    attribute_counts = {name: 0 for name in ATTRIBUTE_NAMES}
    for record in sequence_records:
        category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1
        for name, enabled in record["difficulty_tags"].items():
            if enabled:
                attribute_counts[name] += 1
    for sequence in sequences:
        for frame in sequence.invalid_frames:
            reason = frame.invalid_reason or "unknown"
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1

    train_base_sequences = {record["base_sequence"] for record in sequence_records if record["sequence_name"] in set(train_names)}
    val_base_sequences = {record["base_sequence"] for record in sequence_records if record["sequence_name"] in set(val_names)}
    split_overlap = sorted(train_base_sequences & val_base_sequences)

    summary = {
        "subset": subset,
        "sequence_count": len(sequence_records),
        "train_sequence_count": len(train_names),
        "val_sequence_count": len(val_names),
        "base_sequence_count": len({record["base_sequence"] for record in sequence_records}),
        "valid_frame_count": sum(record["num_valid_frames"] for record in sequence_records),
        "invalid_frame_count": sum(record["num_invalid_frames"] for record in sequence_records),
        "invalid_reason_counts": invalid_reason_counts,
        "category_counts": category_counts,
        "attribute_counts": attribute_counts,
        "train_base_sequence_count": len(train_base_sequences),
        "val_base_sequence_count": len(val_base_sequences),
        "split_overlap_count": len(split_overlap),
        "split_overlap_base_sequences": split_overlap,
        "output_dir": str(paths.root),
    }
    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess UAV123 into manifest/cache/splits")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--subset", type=str, default="UAV123")
    parser.add_argument("--config-filename", type=str, default="configSeqs.m")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    summary = run_preprocess(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        subset=args.subset,
        config_filename=args.config_filename,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("Preprocessing completed successfully:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

