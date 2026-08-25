"""Round-trip unit check: one document per dataset/split.

Asserts token/bbox/label alignment, bbox range/ordering, BIO validity,
image existence/openability, and that split JSON doc-IDs resolve to real
documents (CORD val_es/cal disjointness re-checked from the persisted files).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import BUNDLE_DIR, EXPECTED, hard_stop, status

import pyarrow.parquet as pq
from PIL import Image


def check_doc(dataset, split, row, label_set):
    n = len(row["tokens"])
    assert n == len(row["bboxes"]) == len(row["ner_tags"]), "length misalignment"
    assert n > 0, "empty document"
    if row["ocr_confidence"] is not None:
        assert len(row["ocr_confidence"]) == n, "ocr_confidence misalignment"
    for b in row["bboxes"]:
        assert len(b) == 4 and all(0 <= v <= 1000 for v in b), f"bbox out of range: {b}"
        assert b[0] <= b[2] and b[1] <= b[3], f"bbox unordered: {b}"
    prev = "O"
    for t in row["ner_tags"]:
        assert t in label_set, f"tag {t} not in label list"
        if t.startswith("I-"):
            assert prev != "O" and prev[2:] == t[2:], f"orphan {t} after {prev}"
        prev = t
    img = Image.open(BUNDLE_DIR / row["image_path"])
    img.verify()


def main():
    ok = True
    for dataset, splits in EXPECTED.items():
        labels = set(json.loads(
            (BUNDLE_DIR / dataset / "processed" / "label_list.json").read_text()
        )["labels"])
        for split in splits:
            table = pq.read_table(BUNDLE_DIR / dataset / "processed" / f"{split}.parquet")
            row = table.slice(0, 1).to_pylist()[0]
            try:
                check_doc(dataset, split, row, labels)
                ok &= status(dataset, f"roundtrip {split}", "aligned", row["doc_id"], True)
            except AssertionError as e:
                ok &= status(dataset, f"roundtrip {split}", "aligned", str(e), False)

    # Split JSONs resolve to real doc_ids.
    all_ids = {
        ds: {i for split in sp
             for i in pq.read_table(BUNDLE_DIR / ds / "processed" / f"{split}.parquet",
                                    columns=["doc_id"]).column("doc_id").to_pylist()}
        for ds, sp in EXPECTED.items()
    }
    for sj in sorted((BUNDLE_DIR / "splits").glob("*.json")):
        d = json.loads(sj.read_text())
        ids = d.get("doc_ids", []) + d.get("calibration", []) + d.get("evaluation", [])
        missing = [i for i in ids if i not in all_ids[d["dataset"]]]
        ok &= status(d["dataset"], f"resolve {sj.name}", "0 missing", len(missing), not missing)

    ves = set(json.loads((BUNDLE_DIR / "splits" / "cord_val_es.json").read_text())["doc_ids"])
    cal = set(json.loads((BUNDLE_DIR / "splits" / "cord_cal.json").read_text())["doc_ids"])
    ok &= status("cord", "val_es ∩ cal", "empty", len(ves & cal), not (ves & cal))

    if not ok:
        hard_stop("round-trip checks failed")


if __name__ == "__main__":
    main()
