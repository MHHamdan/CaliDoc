"""Shared helpers for the CaliDoc dataset preparation pipeline."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
BUNDLE_DIR = ROOT / "data_bundle"

FUNSD_LABELS = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-QUESTION", "I-QUESTION",
    "B-ANSWER", "I-ANSWER",
]

EXPECTED = {
    "funsd": {"train": 149, "test": 50},
    "cord": {"train": 800, "validation": 100, "test": 100},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_bbox(box, width, height):
    """Absolute [x1,y1,x2,y2] -> 0-1000 ints, clamped, ordered."""
    x1 = max(0, min(1000, int(1000 * box[0] / width)))
    y1 = max(0, min(1000, int(1000 * box[1] / height)))
    x2 = max(0, min(1000, int(1000 * box[2] / width)))
    y2 = max(0, min(1000, int(1000 * box[3] / height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def status(dataset, step, expected, actual, ok):
    line = f"{dataset:<8} | {step:<22} | expected {expected!s:<14} | actual {actual!s:<14} | {'PASS' if ok else 'FAIL'}"
    print(line, flush=True)
    return ok


def hard_stop(msg):
    print(f"HARD STOP: {msg}", flush=True)
    raise SystemExit(1)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
