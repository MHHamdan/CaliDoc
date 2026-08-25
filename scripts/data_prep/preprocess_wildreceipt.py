"""WildReceipt -> Phase 2 bundle addendum (PI-approved 2026-07-05).

WildReceipt (Sun et al. 2021, arXiv:2103.14470): 1267 train / 472 test
receipts with NATIVE box-level class annotations (box quad + text + class id)
— no entity-string alignment step exists, so the SROIE gate failure mode is
structurally impossible. This script only verifies schema and projects to BIO.

Label policy (PI spec):
- Class names read from class_list.txt (NEVER hardcoded).
- 'Others' -> O; 'Ignore' boxes EXCLUDED with per-split counts recorded.
- Remaining semantic classes x B/I + O -> label_list.json (derived).

Output: data_bundle/wildreceipt/{processed,images}, splits
(wildreceipt_val_es.json seed 42; wildreceipt_calib_seed{42,123,456}.json
70/30 of test), stats, checksums, PROVENANCE addendum, addendum tar.
Existing verified bundle files are NOT modified (addendum only).
"""

import argparse
import datetime
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BUNDLE_DIR = Path(__file__).resolve().parents[2] / "data_bundle"

EXPECTED_COUNTS = {"train": 1267, "test": 472}
VAL_ES_SIZE = 100          # ~8% of train; between CORD (50/800) and FUNSD (15/149)
VAL_ES_SEED = 42
CALIB_SEEDS = (42, 123, 456)
CALIB_RATIO = 0.7          # FUNSD-style document-level 70/30 calibration/evaluation


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_classes(src: Path):
    """id -> stripped class name from class_list.txt (tab/space tolerant)."""
    id2name = {}
    for line in (src / "class_list.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        idx, name = line.split(None, 1)
        id2name[int(idx)] = name.strip()
    assert "Ignore" in id2name.values() and "Others" in id2name.values(), \
        "class_list.txt must contain Ignore and Others"
    return id2name


def norm_bbox(quad, width, height):
    xs, ys = quad[0::2], quad[1::2]
    x1 = max(0, min(1000, int(1000 * min(xs) / width)))
    y1 = max(0, min(1000, int(1000 * min(ys) / height)))
    x2 = max(0, min(1000, int(1000 * max(xs) / width)))
    y2 = max(0, min(1000, int(1000 * max(ys) / height)))
    return [x1, y1, x2, y2]


def process_split(src: Path, split: str, id2name, label_set):
    rows, stats = [], Counter()
    seen_ids = set()
    for line in (src / f"{split}.txt").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        w, h = rec["width"], rec["height"]
        stem = Path(rec["file_name"]).stem
        doc_id = f"wildreceipt_{split}_{stem}"
        if doc_id in seen_ids:
            raise SystemExit(f"duplicate doc_id {doc_id}")
        seen_ids.add(doc_id)

        tokens, bboxes, tags = [], [], []
        for ann in rec["annotations"]:
            name = id2name[ann["label"]]
            if name == "Ignore":
                stats["ignore_boxes"] += 1
                stats["ignore_tokens"] += len(ann["text"].split())
                continue
            words = ann["text"].split()
            if not words:
                stats["empty_text_boxes"] += 1
                continue
            box = norm_bbox(ann["box"], w, h)
            for i, word in enumerate(words):
                tokens.append(word)
                bboxes.append(box)
                if name == "Others":
                    tags.append("O")
                else:
                    tags.append(("B-" if i == 0 else "I-") + name)
        if not tokens:
            raise SystemExit(f"{doc_id}: zero tokens after filtering")
        bad = set(tags) - label_set
        if bad:
            raise SystemExit(f"{doc_id}: tags outside derived label list: {bad}")
        stats["tokens"] += len(tokens)
        stats["o_tokens"] += sum(t == "O" for t in tags)

        rel_img = f"wildreceipt/images/{split}/{doc_id}{Path(rec['file_name']).suffix}"
        rows.append({
            "doc_id": doc_id,
            "src_image": src / rec["file_name"],
            "rel_img": rel_img,
            "tokens": tokens, "bboxes": bboxes, "ner_tags": tags,
        })
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="extracted mirror root containing class_list.txt/train.txt/test.txt/image_files")
    ap.add_argument("--mirror-b-zip", type=Path, default=None,
                    help="second mirror zip (provenance sha256 only)")
    ap.add_argument("--mirror-a-zip", type=Path, default=None)
    args = ap.parse_args()
    src = args.src

    id2name = load_classes(src)
    semantic = sorted(n for n in id2name.values() if n not in ("Ignore", "Others"))
    labels = sorted([f"{p}-{n}" for n in semantic for p in ("B", "I")]) + []
    labels = sorted(labels) + ["O"]
    labels = sorted(set(labels))  # deterministic order: B-*/I-* alphabetical + O
    label_set = set(labels)
    print(f"classes: {len(id2name)} raw -> {len(semantic)} semantic -> {len(labels)} BIO labels")

    out_dir = BUNDLE_DIR / "wildreceipt" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("doc_id", pa.string()),
        ("tokens", pa.list_(pa.string())),
        ("bboxes", pa.list_(pa.list_(pa.int64(), 4))),
        ("ner_tags", pa.list_(pa.string())),
        ("image_path", pa.string()),
        ("split", pa.string()),
        ("ocr_confidence", pa.list_(pa.float64())),
    ])

    split_stats = {}
    all_rows = {}
    for split in ("train", "test"):
        rows, stats = process_split(src, split, id2name, label_set)
        if len(rows) != EXPECTED_COUNTS[split]:
            raise SystemExit(f"{split}: {len(rows)} docs != expected {EXPECTED_COUNTS[split]}")
        img_out = BUNDLE_DIR / "wildreceipt" / "images" / split
        img_out.mkdir(parents=True, exist_ok=True)
        out_rows = []
        for r in rows:
            shutil.copy2(r["src_image"], BUNDLE_DIR / r["rel_img"])
            out_rows.append({
                "doc_id": r["doc_id"], "tokens": r["tokens"], "bboxes": r["bboxes"],
                "ner_tags": r["ner_tags"], "image_path": r["rel_img"],
                "split": split, "ocr_confidence": None,
            })
        pq.write_table(pa.Table.from_pylist(out_rows, schema=schema),
                       out_dir / f"{split}.parquet")
        split_stats[split] = {
            "docs": len(rows),
            "tokens": stats["tokens"],
            "o_fraction": stats["o_tokens"] / stats["tokens"],
            "ignore_boxes_excluded": stats["ignore_boxes"],
            "ignore_tokens_excluded": stats["ignore_tokens"],
            "empty_text_boxes_dropped": stats["empty_text_boxes"],
        }
        all_rows[split] = rows
        print(f"[{split}] {split_stats[split]}")

    with open(out_dir / "label_list.json", "w") as f:
        json.dump({
            "labels": labels,
            "num_bio_labels": len(labels),
            "num_fields": len(semantic),
            "policy": "Others->O; Ignore boxes excluded (counts in stats.json); "
                      "derived from class_list.txt, never hardcoded",
            "expected_o_fraction": {s: split_stats[s]["o_fraction"] for s in split_stats},
        }, f, indent=2)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    # --- splits (persisted JSONs, same formats as CORD/FUNSD) ---
    splits_dir = BUNDLE_DIR / "splits"
    train_ids = sorted(r["doc_id"] for r in all_rows["train"])
    test_ids = sorted(r["doc_id"] for r in all_rows["test"])
    val_es = sorted(random.Random(VAL_ES_SEED).sample(train_ids, VAL_ES_SIZE))
    with open(splits_dir / "wildreceipt_val_es.json", "w") as f:
        json.dump({
            "dataset": "wildreceipt", "source_split": "train", "seed": VAL_ES_SEED,
            "purpose": "early stopping ONLY; these docs must be REMOVED from training",
            "effective_train_size": len(train_ids) - VAL_ES_SIZE,
            "doc_ids": val_es,
        }, f, indent=2)
    n_cal = int(len(test_ids) * CALIB_RATIO)
    for seed in CALIB_SEEDS:
        shuffled = test_ids[:]
        random.Random(seed).shuffle(shuffled)
        with open(splits_dir / f"wildreceipt_calib_seed{seed}.json", "w") as f:
            json.dump({
                "dataset": "wildreceipt", "source_split": "test", "seed": seed,
                "purpose": "document-level 70/30 conformal calibration/evaluation split",
                "calibration": sorted(shuffled[:n_cal]),
                "evaluation": sorted(shuffled[n_cal:]),
            }, f, indent=2)
    print(f"splits: val_es={VAL_ES_SIZE} (train {len(train_ids)}->{len(train_ids)-VAL_ES_SIZE}), "
          f"calib/eval={n_cal}/{len(test_ids)-n_cal} x seeds {CALIB_SEEDS}")

    # --- checksums + provenance + addendum tar ---
    today = datetime.date.today().strftime("%Y%m%d")
    checks = {}
    for p in sorted(out_dir.glob("*")) + [splits_dir / "wildreceipt_val_es.json"] + \
             [splits_dir / f"wildreceipt_calib_seed{s}.json" for s in CALIB_SEEDS]:
        checks[str(p.relative_to(BUNDLE_DIR))] = sha256(p)
    (BUNDLE_DIR / "checksums").mkdir(exist_ok=True)
    with open(BUNDLE_DIR / "checksums" / f"wildreceipt_{today}.json", "w") as f:
        json.dump(checks, f, indent=2)

    addendum = BUNDLE_DIR / f"PROVENANCE_wildreceipt_addendum_{today}.md"
    with open(addendum, "w") as f:
        f.write(f"# WildReceipt addendum ({today}) — PI-approved third dataset\n\n")
        f.write("Source: WildReceipt (Sun et al. 2021, arXiv:2103.14470), native\n"
                "box-level annotations — no entity-string alignment step exists.\n\n")
        f.write("## Provenance\n")
        f.write("Canonical host download.openmmlab.com was UNAVAILABLE (DNS SERVFAIL\n"
                "upstream) at retrieval time. Two independent Kaggle mirrors used,\n"
                "byte-identical on all annotation files (train.txt, test.txt,\n"
                "class_list.txt, dict.txt), identical 1769-image name sets, 30/30\n"
                "random image hashes identical:\n")
        if args.mirror_a_zip and args.mirror_a_zip.exists():
            f.write(f"- mirror A `shregar/doc-wildreceipt`: sha256 `{sha256(args.mirror_a_zip)}`\n")
        if args.mirror_b_zip and args.mirror_b_zip.exists():
            f.write(f"- mirror B `maazahmad69/wild-receipt`: sha256 `{sha256(args.mirror_b_zip)}`\n")
        f.write("\n## Label policy\n- Derived from class_list.txt (never hardcoded): "
                f"{len(semantic)} semantic classes x B/I + O = {len(labels)} BIO labels\n"
                "- Others -> O; Ignore boxes EXCLUDED:\n")
        for s, st in split_stats.items():
            f.write(f"  - {s}: {st['ignore_boxes_excluded']} boxes / "
                    f"{st['ignore_tokens_excluded']} tokens excluded; "
                    f"O fraction {st['o_fraction']:.4%}\n")
        f.write(f"\n## Splits\n- official train/test preserved: "
                f"{EXPECTED_COUNTS['train']}/{EXPECTED_COUNTS['test']}\n"
                f"- val_es: {VAL_ES_SIZE} docs from train, seed {VAL_ES_SEED} (persisted)\n"
                f"- conformal: 70/30 calib/eval of test, seeds {CALIB_SEEDS} (persisted)\n")
        f.write("\n## Note\nSROIE remains EXCLUDED (failed pre-registered exact-match "
                "gate, 94.68%/94.02% < 95%); it is a data-quality gate result, not an "
                "experimental dataset. WildReceipt takes its experiment slot.\n")

    tar_path = BUNDLE_DIR.parent / f"wildreceipt_addendum_{today}.tar"
    root = BUNDLE_DIR.name
    subprocess.run(
        ["tar", "-cf", str(tar_path), "-C", str(BUNDLE_DIR.parent),
         f"{root}/wildreceipt",
         f"{root}/splits/wildreceipt_val_es.json",
         *[f"{root}/splits/wildreceipt_calib_seed{s}.json" for s in CALIB_SEEDS],
         f"{root}/checksums/wildreceipt_{today}.json",
         str(addendum.relative_to(BUNDLE_DIR.parent))],
        check=True)
    print(f"\nWrote {tar_path}\nsha256: {sha256(tar_path)}")


if __name__ == "__main__":
    main()
