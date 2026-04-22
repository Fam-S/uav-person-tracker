import csv
from pathlib import Path


TEMPLATE_PATH = Path("metadata") / "sample_submission.csv"


def _read_template(raw_root):
    """Open the submission template CSV and return its header + all rows."""
    path = Path(raw_root) / TEMPLATE_PATH
    with path.open("r", encoding="utf-8", newline="") as f:
        # DictReader treats the first row as column names, so each later row
        # becomes a dict like: {"id": "dataset1/Car_video_0", "x": "0", ...}
        reader = csv.DictReader(f)
        # fieldnames is the header list: ["id", "x", "y", "w", "h"]
        return reader.fieldnames, list(reader)


def load_submission_ids(raw_root):
    """Return template row IDs in the exact submission order."""
    _, rows = _read_template(raw_root)
    return [row["id"] for row in rows]


def write_submission_csv(raw_root, output_path, predictions):
    """Write a submission CSV using template order and provided bbox predictions."""
    fieldnames, rows = _read_template(raw_root)
    output_path = Path(output_path)

    # Create any missing parent directories before opening the output file.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as dst:
        # DictWriter writes dict rows using the same header names/order.
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            # Each template row has the exact submission `id` we must keep.
            # If a prediction is missing for this id, fall back to 0,0,0,0.
            bbox = predictions.get(row["id"], (0, 0, 0, 0))
            writer.writerow(
                {
                    "id": row["id"],
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "w": int(bbox[2]),
                    "h": int(bbox[3]),
                }
            )
