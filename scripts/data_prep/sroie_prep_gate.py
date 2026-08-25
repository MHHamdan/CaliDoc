"""SROIE Task 3 (ICDAR 2019 KIE) preparation with a hard alignment gate.

SROIE annotations are entity STRINGS (company, date, address, total), not
token tags. This script aligns each entity string to the receipt's OCR token
sequence by EXACT match — the only normalization allowed is whitespace
(collapsed/removed) and case. No fuzzy matching, no edit-distance rescue.

GATE: if <95% of entities align exactly in any split, the script stops,
prints the failure distribution, and writes NOTHING to the bundle. SROIE is
then dropped from the paper — do not hand-patch alignments.

On pass, emits per split data_bundle/sroie/processed/<split>.parquet with the
FUNSD/CORD schema {doc_id, tokens, bboxes (0-1000), ner_tags (BIO strings),
image_path, split, ocr_confidence(null)}, 9 BIO labels (4 fields x B/I + O),
a persisted 60-doc val_es carve (seed 42), stats, checksums, a PROVENANCE
addendum, and sroie_addendum_YYYYMMDD.tar for transfer.

Expected input (downloaded from the ICDAR RRC SROIE portal, user account
required):
  --train-ocr       dir with per-receipt OCR files X.txt
                    (lines: x1,y1,...,x4,y4,transcript) and images X.jpg
  --train-entities  dir with per-receipt entity JSON files X.txt
  --test-ocr / --test-entities  same for the 347-receipt test split
  --source-zips     the original downloaded archives (sha256'd into PROVENANCE)

Usage:
  python scripts/data_prep/sroie_prep_gate.py \
      --train-ocr data/raw/sroie/train_ocr --train-entities data/raw/sroie/train_entities \
      --test-ocr data/raw/sroie/test_ocr --test-entities data/raw/sroie/test_entities \
      --source-zips data/raw/sroie/*.zip
"""

import argparse
import datetime
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BUNDLE_DIR = Path(__file__).resolve().parents[2] / "data_bundle"

FIELDS = ["company", "date", "address", "total"]
SROIE_LABELS = ["O"] + [f"{p}-{f.upper()}" for f in FIELDS for p in ("B", "I")]
assert len(SROIE_LABELS) == 9

EXPECTED_COUNTS = {"train": 626, "test": 347}
GATE_THRESHOLD = 0.95
VAL_ES_SIZE = 60
VAL_ES_SEED = 42


def norm_chars(s: str) -> str:
    """The ONLY allowed normalization: strip all whitespace, casefold."""
    return re.sub(r"\s+", "", s).casefold()


def parse_ocr_file(path: Path):
    """Parse SROIE OCR file: x1,y1,x2,y2,x3,y3,x4,y4,transcript per line.

    Returns word-level tokens; each word inherits its line's box (SROIE boxes
    are line-level).
    """
    tokens, boxes = [], []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",", 8)
        if len(parts) < 9:
            continue  # malformed line; counted implicitly via alignment gate
        try:
            coords = [int(float(v)) for v in parts[:8]]
        except ValueError:
            continue
        text = parts[8]
        xs, ys = coords[0::2], coords[1::2]
        box = [min(xs), min(ys), max(xs), max(ys)]
        for word in text.split():
            tokens.append(word)
            boxes.append(box)
    return tokens, boxes


def find_span(doc_tokens_norm, entity: str):
    """Find the first contiguous token span whose concatenated normalized
    chars equal the normalized entity string. Returns (start, end) inclusive
    token indices, or None."""
    target = norm_chars(entity)
    if not target:
        return None
    n = len(doc_tokens_norm)
    tlen = len(target)
    for start in range(n):
        # Skip empty-normalized tokens at start
        if not doc_tokens_norm[start]:
            continue
        acc = 0
        for end in range(start, n):
            tok = doc_tokens_norm[end]
            seg = target[acc:acc + len(tok)]
            if tok != seg:
                break
            acc += len(tok)
            if acc == tlen:
                return (start, end)
            if acc > tlen:
                break
    return None


def normalize_bbox(box, width, height):
    x1 = max(0, min(1000, int(1000 * box[0] / width)))
    y1 = max(0, min(1000, int(1000 * box[1] / height)))
    x2 = max(0, min(1000, int(1000 * box[2] / width)))
    y2 = max(0, min(1000, int(1000 * box[3] / height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_split(split, ocr_dir: Path, ent_dir: Path):
    """Align every receipt; returns (rows, alignment_stats). No bundle writes."""
    ocr_files = sorted(ocr_dir.glob("*.txt"))
    stats = {
        "receipts": 0,
        "entities_total": 0,
        "entities_aligned": 0,
        "failures_by_field": Counter(),
        "failure_examples": defaultdict(list),
    }
    rows = []
    for ocr_path in ocr_files:
        stem = ocr_path.stem
        ent_path = ent_dir / f"{stem}.txt"
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = ocr_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if not ent_path.exists() or img_path is None:
            print(f"  WARN {split}/{stem}: missing entities or image, skipped receipt")
            continue

        stats["receipts"] += 1
        tokens, boxes = parse_ocr_file(ocr_path)
        entities = json.loads(ent_path.read_text(encoding="utf-8", errors="replace"))
        tokens_norm = [norm_chars(t) for t in tokens]

        tags = ["O"] * len(tokens)
        for f in FIELDS:
            value = entities.get(f, "")
            if not str(value).strip():
                continue  # absent field: not counted in the denominator
            stats["entities_total"] += 1
            span = find_span(tokens_norm, str(value))
            if span is None:
                stats["failures_by_field"][f] += 1
                if len(stats["failure_examples"][f]) < 5:
                    stats["failure_examples"][f].append({"doc": stem, "value": str(value)[:80]})
                continue
            stats["entities_aligned"] += 1
            s, e = span
            tags[s] = f"B-{f.upper()}"
            for i in range(s + 1, e + 1):
                tags[i] = f"I-{f.upper()}"

        rows.append({
            "stem": stem, "img_src": img_path,
            "tokens": tokens, "boxes": boxes, "tags": tags,
        })
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ocr", required=True, type=Path)
    ap.add_argument("--train-entities", required=True, type=Path)
    ap.add_argument("--test-ocr", required=True, type=Path)
    ap.add_argument("--test-entities", required=True, type=Path)
    ap.add_argument("--source-zips", nargs="*", type=Path, default=[])
    args = ap.parse_args()

    # ---- Phase A: align everything, evaluate the gate, write NOTHING yet ----
    aligned = {}
    all_stats = {}
    gate_ok = True
    for split, ocr_dir, ent_dir in [
        ("train", args.train_ocr, args.train_entities),
        ("test", args.test_ocr, args.test_entities),
    ]:
        rows, stats = process_split(split, ocr_dir, ent_dir)
        rate = stats["entities_aligned"] / max(1, stats["entities_total"])
        stats["alignment_rate"] = rate
        aligned[split] = rows
        all_stats[split] = stats
        print(f"[{split}] receipts={stats['receipts']} (expected {EXPECTED_COUNTS[split]}) "
              f"entities={stats['entities_total']} aligned={stats['entities_aligned']} "
              f"rate={rate:.4%}")
        if stats["receipts"] != EXPECTED_COUNTS[split]:
            print(f"  WARN: receipt count differs from expected {EXPECTED_COUNTS[split]}")
        if rate < GATE_THRESHOLD:
            gate_ok = False

    if not gate_ok:
        print("\n" + "=" * 70)
        print("GATE FAILED: <95% exact alignment. SROIE is DROPPED from the paper.")
        print("Do NOT hand-patch. Failure distribution:")
        for split, stats in all_stats.items():
            print(f"\n[{split}] rate={stats['alignment_rate']:.4%}")
            for f, cnt in stats["failures_by_field"].most_common():
                print(f"  {f}: {cnt} failures")
                for ex in stats["failure_examples"][f]:
                    print(f"    e.g. {ex['doc']}: {ex['value']!r}")
        sys.exit(1)

    print(f"\nGATE PASSED (>= {GATE_THRESHOLD:.0%} in every split). Emitting bundle files.")

    # ---- Phase B: emit parquet + images + splits + provenance + tar ----
    out_dir = BUNDLE_DIR / "sroie" / "processed"
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
    for split, rows in aligned.items():
        img_out = BUNDLE_DIR / "sroie" / "images" / split
        img_out.mkdir(parents=True, exist_ok=True)
        out_rows = []
        for r in rows:
            width, height = Image.open(r["img_src"]).size
            ext = r["img_src"].suffix.lower()
            doc_id = f"sroie_{split}_{r['stem']}"
            rel_img = f"sroie/images/{split}/{doc_id}{ext}"
            shutil.copy2(r["img_src"], BUNDLE_DIR / rel_img)
            out_rows.append({
                "doc_id": doc_id,
                "tokens": r["tokens"],
                "bboxes": [normalize_bbox(b, width, height) for b in r["boxes"]],
                "ner_tags": r["tags"],
                "image_path": rel_img,
                "split": split,
                "ocr_confidence": None,  # SROIE provides no OCR confidence
            })
        table = pa.Table.from_pylist(out_rows, schema=schema)
        pq.write_table(table, out_dir / f"{split}.parquet")
        n_tok = sum(len(r["tokens"]) for r in rows)
        n_o = sum(t == "O" for r in rows for t in r["tags"])
        split_stats[split] = {
            "docs": len(rows), "tokens": n_tok,
            "o_fraction": n_o / max(1, n_tok),
            "alignment_rate": all_stats[split]["alignment_rate"],
        }

    with open(out_dir / "label_list.json", "w") as f:
        json.dump({"labels": SROIE_LABELS, "num_bio_labels": 9, "num_fields": 4}, f, indent=2)

    # 60-doc val_es carve from train, seed 42, persisted
    train_ids = sorted(f"sroie_train_{r['stem']}" for r in aligned["train"])
    val_es = sorted(random.Random(VAL_ES_SEED).sample(train_ids, VAL_ES_SIZE))
    (BUNDLE_DIR / "splits").mkdir(exist_ok=True)
    with open(BUNDLE_DIR / "splits" / "sroie_val_es.json", "w") as f:
        json.dump({
            "dataset": "sroie", "source_split": "train", "seed": VAL_ES_SEED,
            "purpose": "early stopping ONLY; these docs must be REMOVED from training",
            "effective_train_size": len(train_ids) - VAL_ES_SIZE,
            "doc_ids": val_es,
        }, f, indent=2)

    # Checksums + provenance addendum
    today = datetime.date.today().strftime("%Y%m%d")
    checks = {}
    for p in sorted(out_dir.glob("*.parquet")) + [out_dir / "label_list.json",
                                                  BUNDLE_DIR / "splits" / "sroie_val_es.json"]:
        checks[str(p.relative_to(BUNDLE_DIR))] = sha256(p)
    src_checks = {str(p): sha256(p) for p in args.source_zips}
    (BUNDLE_DIR / "checksums").mkdir(exist_ok=True)
    with open(BUNDLE_DIR / "checksums" / f"sroie_{today}.json", "w") as f:
        json.dump({"bundle_files": checks, "source_archives": src_checks}, f, indent=2)

    addendum = BUNDLE_DIR / f"PROVENANCE_sroie_addendum_{today}.md"
    with open(addendum, "w") as f:
        f.write(f"# SROIE Task 3 addendum ({today})\n\n")
        f.write("Source: ICDAR 2019 RRC SROIE (Task 3 KIE), downloaded by user "
                "from rrc.cvc.uab.es (account-gated).\n\n## Source archives\n")
        for p, h in src_checks.items():
            f.write(f"- `{p}`: sha256 `{h}`\n")
        f.write("\n## Alignment (exact match, whitespace/case normalization only)\n")
        for split, st in split_stats.items():
            f.write(f"- {split}: {st['docs']} receipts, alignment rate "
                    f"{st['alignment_rate']:.4%}, O fraction {st['o_fraction']:.4%}\n")
        f.write(f"\n## Splits\n- val_es: {VAL_ES_SIZE} docs from train, seed {VAL_ES_SEED}, "
                f"persisted splits/sroie_val_es.json; effective train "
                f"{len(train_ids) - VAL_ES_SIZE}\n")
        f.write("\n## Labels\n- 9 BIO labels: 4 fields (company, date, address, total) "
                "x B/I + O\n")
        f.write("\n## Transfer\n- TODO record: Globus or rsync (<1GB acceptable), "
                "fill in after transfer.\n")

    tar_path = BUNDLE_DIR.parent / f"sroie_addendum_{today}.tar"
    root = BUNDLE_DIR.name
    subprocess.run(
        ["tar", "-cf", str(tar_path), "-C", str(BUNDLE_DIR.parent),
         f"{root}/sroie", f"{root}/splits/sroie_val_es.json",
         f"{root}/checksums/sroie_{today}.json",
         str(addendum.relative_to(BUNDLE_DIR.parent))],
        check=True,
    )
    print(f"\nWrote {tar_path}  sha256={sha256(tar_path)}")
    print(json.dumps(split_stats, indent=2))
    print("Next: sroie_prep_gate PASSED -> proceed to sroie_retrain_s42 (Trillium).")


if __name__ == "__main__":
    main()
