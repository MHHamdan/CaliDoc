"""SROIE via UPSTREAM PRE-TAGGED BIO -> bundle addendum.

*** PENDING PI SIGN-OFF (2026-07-07) ***
This reverses the standing exclusion of SROIE (which FAILED our pre-registered
exact-match gate at 94.68%/94.02% < 95%). It is built at the user's request as
an ADDITIONAL, clearly-captioned benchmark using the community-standard
pre-tagged SROIE BIO (the `darentang/sroie` distribution), NOT our own
alignment. The token labels here were fuzzy-aligned UPSTREAM by the dataset
authors; roughly 5-6% of entity tokens do not exactly match the OCR text
(quantified by our gate). The gate report (results/sroie_gate_report_20260705.md)
is retained. Every SROIE number MUST be reported with the label-provenance
caveat below. Do not merge into the paper until the PI approves.

Output mirrors WildReceipt: data_bundle/sroie/{processed,images}, splits
(sroie_val_es.json seed 42; sroie_calib_seed{42,123,456}.json 70/30 of test),
label_list.json (with provenance caveat), stats, checksums, PROVENANCE
addendum, addendum tar. Existing verified files are NOT modified.
"""

import datetime
import hashlib
import io
import json
import random
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BUNDLE_DIR = Path(__file__).resolve().parents[2] / "data_bundle"
SROIE_ZIP = BUNDLE_DIR.parent / "data" / "raw" / "sroie_gdrive" / "sroie.zip"

EXPECTED_COUNTS = {"train": 626, "test": 347}
VAL_ES_SIZE = 60
VAL_ES_SEED = 42
CALIB_SEEDS = (42, 123, 456)
CALIB_RATIO = 0.7


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_bbox(b, w, h):
    x1 = max(0, min(1000, int(1000 * b[0] / w)))
    y1 = max(0, min(1000, int(1000 * b[1] / h)))
    x2 = max(0, min(1000, int(1000 * b[2] / w)))
    y2 = max(0, min(1000, int(1000 * b[3] / h)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def main():
    z = zipfile.ZipFile(SROIE_ZIP)
    names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
    label_set = set()
    out_dir = BUNDLE_DIR / "sroie" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("doc_id", pa.string()), ("tokens", pa.list_(pa.string())),
        ("bboxes", pa.list_(pa.list_(pa.int64(), 4))), ("ner_tags", pa.list_(pa.string())),
        ("image_path", pa.string()), ("split", pa.string()),
        ("ocr_confidence", pa.list_(pa.float64())),
    ])

    split_stats, all_ids = {}, {}
    for split in ("train", "test"):
        tagged = sorted(n for n in names if f"/{split}/tagged/" in n and n.endswith(".json"))
        img_out = BUNDLE_DIR / "sroie" / "images" / split
        img_out.mkdir(parents=True, exist_ok=True)
        rows = []
        stats = Counter()
        for n in tagged:
            stem = n.split("/")[-1][:-5]
            doc_id = f"sroie_{split}_{stem}"
            d = json.loads(z.read(n))
            img_entry = next((m for m in names if f"/{split}/images/{stem}." in m), None)
            if img_entry is None:
                raise SystemExit(f"missing image for {stem}")
            im = Image.open(io.BytesIO(z.read(img_entry))).convert("RGB")
            w, h = im.size
            tokens = [t for t in d["words"]]
            tags = list(d["labels"])
            boxes = [norm_bbox(b, w, h) for b in d["bbox"]]
            if not (len(tokens) == len(tags) == len(boxes)):
                raise SystemExit(f"{doc_id}: length mismatch")
            label_set.update(tags)
            stats["tokens"] += len(tokens)
            stats["o_tokens"] += sum(t == "O" for t in tags)
            rel_img = f"sroie/images/{split}/{doc_id}.jpg"
            im.save(BUNDLE_DIR / rel_img, "JPEG")
            rows.append({"doc_id": doc_id, "tokens": tokens, "bboxes": boxes,
                         "ner_tags": tags, "image_path": rel_img, "split": split,
                         "ocr_confidence": None})
        if len(rows) != EXPECTED_COUNTS[split]:
            raise SystemExit(f"{split}: {len(rows)} != {EXPECTED_COUNTS[split]}")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_dir / f"{split}.parquet")
        split_stats[split] = {"docs": len(rows), "tokens": stats["tokens"],
                              "o_fraction": stats["o_tokens"] / stats["tokens"]}
        all_ids[split] = sorted(r["doc_id"] for r in rows)
        print(f"[{split}] {split_stats[split]}")

    labels = sorted(label_set)
    assert len(labels) == 9, labels
    with open(out_dir / "label_list.json", "w") as f:
        json.dump({
            "labels": labels, "num_bio_labels": len(labels), "num_fields": 4,
            "expected_o_fraction": {s: split_stats[s]["o_fraction"] for s in split_stats},
            "label_provenance": "UPSTREAM PRE-TAGGED (darentang/sroie); token labels "
                "fuzzy-aligned by dataset authors. ~5-6% of entity tokens are NOT exact "
                "OCR substrings (our pre-registered gate: 94.68%/94.02% exact < 95%). "
                "Report SROIE numbers with this caveat. PENDING PI SIGN-OFF.",
        }, f, indent=2)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    splits_dir = BUNDLE_DIR / "splits"
    val_es = sorted(random.Random(VAL_ES_SEED).sample(all_ids["train"], VAL_ES_SIZE))
    with open(splits_dir / "sroie_val_es.json", "w") as f:
        json.dump({"dataset": "sroie", "source_split": "train", "seed": VAL_ES_SEED,
                   "purpose": "early stopping ONLY; these docs must be REMOVED from training",
                   "effective_train_size": len(all_ids["train"]) - VAL_ES_SIZE,
                   "doc_ids": val_es}, f, indent=2)
    n_cal = int(len(all_ids["test"]) * CALIB_RATIO)
    for seed in CALIB_SEEDS:
        sh = all_ids["test"][:]
        random.Random(seed).shuffle(sh)
        with open(splits_dir / f"sroie_calib_seed{seed}.json", "w") as f:
            json.dump({"dataset": "sroie", "source_split": "test", "seed": seed,
                       "purpose": "document-level 70/30 conformal calibration/evaluation split",
                       "calibration": sorted(sh[:n_cal]), "evaluation": sorted(sh[n_cal:])},
                      f, indent=2)
    print(f"splits: val_es={VAL_ES_SIZE} (train->{len(all_ids['train'])-VAL_ES_SIZE}), "
          f"calib/eval={n_cal}/{len(all_ids['test'])-n_cal} x {CALIB_SEEDS}")

    today = datetime.date.today().strftime("%Y%m%d")
    checks = {}
    for p in sorted(out_dir.glob("*")) + [splits_dir / "sroie_val_es.json"] + \
             [splits_dir / f"sroie_calib_seed{s}.json" for s in CALIB_SEEDS]:
        checks[str(p.relative_to(BUNDLE_DIR))] = sha256(p)
    (BUNDLE_DIR / "checksums").mkdir(exist_ok=True)
    with open(BUNDLE_DIR / "checksums" / f"sroie_pretagged_{today}.json", "w") as f:
        json.dump(checks, f, indent=2)

    addendum = BUNDLE_DIR / f"PROVENANCE_sroie_pretagged_addendum_{today}.md"
    with open(addendum, "w") as f:
        f.write(f"# SROIE pre-tagged addendum ({today}) — PENDING PI SIGN-OFF\n\n")
        f.write("**This reverses the standing SROIE exclusion. Do not merge into the "
                "paper until the PI approves.**\n\n")
        f.write("## What this is\nSROIE included as an ADDITIONAL benchmark using the "
                "community-standard pre-tagged BIO (`darentang/sroie`, gdrive zip sha256 "
                f"`{sha256(SROIE_ZIP)}`). Labels were fuzzy-aligned UPSTREAM by the dataset "
                "authors, not by us.\n\n")
        f.write("## Mandatory caveat for every SROIE number\nOur pre-registered exact-match "
                "gate found only 94.68% (train) / 94.02% (test) of entity tokens are exact "
                "OCR substrings; the remaining ~5-6% carry upstream fuzzy-alignment label "
                "noise. See results/sroie_gate_report_20260705.md (retained). SROIE is thus "
                "reported separately from the three clean datasets (CORD, FUNSD, "
                "WildReceipt) with this caveat stated.\n\n")
        f.write(f"## Schema/splits\n- {len(labels)} BIO labels (4 fields x B/I + O), from "
                "the pre-tagged data (never hardcoded)\n"
                f"- official {EXPECTED_COUNTS['train']}/{EXPECTED_COUNTS['test']} preserved; "
                f"O fraction {split_stats['train']['o_fraction']:.4%}/"
                f"{split_stats['test']['o_fraction']:.4%}\n"
                f"- val_es {VAL_ES_SIZE} (seed {VAL_ES_SEED}); conformal 70/30 test, "
                f"seeds {CALIB_SEEDS}\n")

    tar_path = BUNDLE_DIR.parent / f"sroie_pretagged_addendum_{today}.tar"
    root = BUNDLE_DIR.name
    subprocess.run(["tar", "-cf", str(tar_path), "-C", str(BUNDLE_DIR.parent),
                    f"{root}/sroie", f"{root}/splits/sroie_val_es.json",
                    *[f"{root}/splits/sroie_calib_seed{s}.json" for s in CALIB_SEEDS],
                    f"{root}/checksums/sroie_pretagged_{today}.json",
                    str(addendum.relative_to(BUNDLE_DIR.parent))], check=True)
    print(f"\nWrote {tar_path}\nsha256: {sha256(tar_path)}")


if __name__ == "__main__":
    main()
