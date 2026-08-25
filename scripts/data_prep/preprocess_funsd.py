"""Preprocess official FUNSD into the normalized CaliDoc format.

Output per split: data_bundle/funsd/processed/<split>.parquet with columns
{doc_id, tokens, bboxes (0-1000), ner_tags (BIO strings), image_path, split,
ocr_confidence} plus original images copied to data_bundle/funsd/images/<split>/.

Conventions match calidoc/data/funsd_dataset.py: word-level boxes, empty-text
words dropped, label 'other' -> O, else B-/I-<LABEL>. FUNSD ships no OCR
confidence -> ocr_confidence is explicitly null.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUNDLE_DIR, EXPECTED, FUNSD_LABELS, RAW_DIR, hard_stop,
                    normalize_bbox, status)

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SPLIT_DIRS = {"train": "training_data", "test": "testing_data"}


def process_split(split):
    src = RAW_DIR / "funsd" / "dataset" / SPLIT_DIRS[split]
    img_out = BUNDLE_DIR / "funsd" / "images" / split
    img_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ann_path in sorted((src / "annotations").glob("*.json")):
        doc_id = ann_path.stem
        img_src = src / "images" / f"{doc_id}.png"
        if not img_src.exists():
            hard_stop(f"missing image for {doc_id}")
        width, height = Image.open(img_src).size

        with open(ann_path) as f:
            form = json.load(f)["form"]

        tokens, bboxes, tags = [], [], []
        for item in form:
            words = [w for w in item["words"] if w["text"].strip()]
            label = item["label"]
            for i, w in enumerate(words):
                tokens.append(w["text"])
                bboxes.append(normalize_bbox(w["box"], width, height))
                if label == "other":
                    tags.append("O")
                else:
                    tags.append(("B-" if i == 0 else "I-") + label.upper())

        bad = set(tags) - set(FUNSD_LABELS)
        if bad:
            hard_stop(f"unexpected FUNSD tags {bad} in {doc_id}")
        if not tokens:
            hard_stop(f"document {doc_id} produced zero tokens")

        rel_img = f"funsd/images/{split}/{doc_id}.png"
        shutil.copy2(img_src, BUNDLE_DIR / rel_img)
        rows.append({
            "doc_id": f"funsd_{split}_{doc_id}",
            "tokens": tokens,
            "bboxes": bboxes,
            "ner_tags": tags,
            "image_path": rel_img,
            "split": split,
            "ocr_confidence": None,  # FUNSD provides no OCR confidence
        })
    return rows


def main():
    schema = pa.schema([
        ("doc_id", pa.string()),
        ("tokens", pa.list_(pa.string())),
        ("bboxes", pa.list_(pa.list_(pa.int64(), 4))),
        ("ner_tags", pa.list_(pa.string())),
        ("image_path", pa.string()),
        ("split", pa.string()),
        ("ocr_confidence", pa.list_(pa.float64())),
    ])

    ok = True
    out_dir = BUNDLE_DIR / "funsd" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, exp in EXPECTED["funsd"].items():
        rows = process_split(split)
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, out_dir / f"{split}.parquet")
        ok &= status("funsd", f"preprocess {split}", exp, len(rows), len(rows) == exp)
    if not ok:
        hard_stop("FUNSD processed counts do not match 149/50")

    from common import write_json
    write_json(out_dir / "label_list.json", {
        "labels": FUNSD_LABELS,
        "num_bio_labels": len(FUNSD_LABELS),
        "num_fields": 3,
    })


if __name__ == "__main__":
    main()
