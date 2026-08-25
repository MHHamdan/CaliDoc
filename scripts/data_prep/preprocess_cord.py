"""Preprocess CORD v2 (from the pinned raw snapshot) into the normalized format.

Output per split: data_bundle/cord/processed/<split>.parquet with columns
{doc_id, source_image_id, tokens, bboxes (0-1000), ner_tags (BIO strings),
image_path, split, ocr_confidence} plus original image bytes written verbatim
(no re-encoding) to data_bundle/cord/images/<split>/.

Conventions match calidoc/data/cord_dataset.py: quad -> [min_x,min_y,max_x,max_y],
line-level BIO (first word of a valid_line gets B-, rest I-), empty-text words
dropped. The label set is DERIVED from the data and reported, not assumed.
CORD words carry no OCR confidence -> ocr_confidence is explicitly null unless
a confidence field is present in the raw annotations.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUNDLE_DIR, EXPECTED, RAW_DIR, hard_stop, normalize_bbox,
                    status, write_json)

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image as HFImage
from datasets import load_dataset


def sniff_ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    hard_stop("unrecognized image format in CORD sample")


def parse_gt(gt: dict):
    tokens, bboxes_abs, tags, confs = [], [], [], []
    any_conf = False
    for line in gt.get("valid_line", []):
        words = [w for w in line.get("words", []) if w.get("text", "").strip()]
        category = line.get("category") or "other"
        for i, w in enumerate(words):
            tokens.append(w["text"])
            q = w["quad"]
            xs = [q["x1"], q["x2"], q["x3"], q["x4"]]
            ys = [q["y1"], q["y2"], q["y3"], q["y4"]]
            bboxes_abs.append([min(xs), min(ys), max(xs), max(ys)])
            if category == "other":
                tags.append("O")
            else:
                tags.append(("B-" if i == 0 else "I-") + category.upper())
            if "confidence" in w:
                any_conf = True
                confs.append(float(w["confidence"]))
            else:
                confs.append(None)
    return tokens, bboxes_abs, tags, (confs if any_conf else None)


def process_split(split, ds):
    img_out = BUNDLE_DIR / "cord" / "images" / split
    img_out.mkdir(parents=True, exist_ok=True)

    rows, label_set = [], set()
    for idx, sample in enumerate(ds):
        gt = json.loads(sample["ground_truth"])
        meta = gt.get("meta", {})
        size = meta.get("image_size", {})
        width, height = size.get("width"), size.get("height")
        if not width or not height:
            hard_stop(f"cord {split}[{idx}]: missing image_size in meta")

        tokens, bboxes_abs, tags, confs = parse_gt(gt)
        if not tokens:
            hard_stop(f"cord {split}[{idx}]: zero tokens")

        img_bytes = sample["image"]["bytes"]
        ext = sniff_ext(img_bytes)
        doc_id = f"cord_{split}_{idx:04d}"
        rel_img = f"cord/images/{split}/{doc_id}.{ext}"
        with open(BUNDLE_DIR / rel_img, "wb") as f:
            f.write(img_bytes)

        label_set.update(tags)
        rows.append({
            "doc_id": doc_id,
            "source_image_id": meta.get("image_id"),
            "tokens": tokens,
            "bboxes": [normalize_bbox(b, width, height) for b in bboxes_abs],
            "ner_tags": tags,
            "image_path": rel_img,
            "split": split,
            "ocr_confidence": confs,
        })
    return rows, label_set


def main():
    snapshot = RAW_DIR / "cord" / "snapshot"
    data_files = {
        split: sorted(str(p) for p in snapshot.rglob(f"{split}-*.parquet"))
        for split in EXPECTED["cord"]
    }
    for split, files in data_files.items():
        if not files:
            hard_stop(f"no raw parquet shards for cord split '{split}'")

    dsd = load_dataset("parquet", data_files=data_files)
    dsd = dsd.cast_column("image", HFImage(decode=False))

    schema = pa.schema([
        ("doc_id", pa.string()),
        ("source_image_id", pa.int64()),
        ("tokens", pa.list_(pa.string())),
        ("bboxes", pa.list_(pa.list_(pa.int64(), 4))),
        ("ner_tags", pa.list_(pa.string())),
        ("image_path", pa.string()),
        ("split", pa.string()),
        ("ocr_confidence", pa.list_(pa.float64())),
    ])

    out_dir = BUNDLE_DIR / "cord" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, all_labels = True, set()
    for split, exp in EXPECTED["cord"].items():
        rows, labels = process_split(split, dsd[split])
        all_labels |= labels
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, out_dir / f"{split}.parquet")
        ok &= status("cord", f"preprocess {split}", exp, len(rows), len(rows) == exp)
    if not ok:
        hard_stop("CORD processed counts do not match 800/100/100")

    label_list = sorted(all_labels)
    n_fields = len({t[2:] for t in label_list if t != "O"})
    write_json(BUNDLE_DIR / "cord" / "processed" / "label_list.json", {
        "labels": label_list,
        "num_bio_labels": len(label_list),
        "num_fields": n_fields,
    })
    status("cord", "derive label set", "from data", f"{len(label_list)} BIO / {n_fields} fields", True)


if __name__ == "__main__":
    main()
