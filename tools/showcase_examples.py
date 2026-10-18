#!/usr/bin/env python3
"""Run the released CaliDoc models on deterministically selected benchmark
documents and write machine-generated prediction JSON.

Selection rule (fixed before any prediction was inspected)
---------------------------------------------------------
Eligible documents are the evaluation side of the released calibration unit:
the CORD test split (CORD calibrates on validation documents), and the
`evaluation` list of `<ds>_calib_seed42.json` for FUNSD, WildReceipt and SROIE,
so no shown document was used to fit a threshold. Eligible documents are
sorted by (word count, doc_id) and we take

* representative: the lower median, index (n - 1) // 2;
* diagnostic (CORD only): the 90th-percentile word count, index
  floor(0.9 * (n - 1)), a dense receipt that exercises CORD's long-tail
  schema and pooled fallback.

Redistribution
--------------
Only CORD (CC BY 4.0) is cleared for redistribution (docs/release/
DATASET_LICENSE_AUDIT.md). For FUNSD, WildReceipt and SROIE the written JSON
keeps field types, word positions, set sizes, thresholds and statuses but drops
words, boxes and images; run with --full locally on your own copy of the
dataset to get the complete output.

    python tools/showcase_examples.py --model_root MHamdan/ --out assets/examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from calidoc.inference import CaliDocPredictor  # noqa: E402

CLEARED = {"cord"}
ATTRIBUTION = {
    "cord": {
        "dataset": "CORD v2",
        "source": "https://huggingface.co/datasets/naver-clova-ix/cord-v2",
        "revision": "7f0115a4b758a71d6473b8d085751692da2fef98",
        "license": "CC BY 4.0",
        "citation": "Park et al., CORD: A Consolidated Receipt Dataset for Post-OCR Parsing, "
                    "Document Intelligence Workshop at NeurIPS 2019",
    },
    "funsd": {
        "dataset": "FUNSD",
        "source": "https://guillaumejaume.github.io/FUNSD/",
        "license": "FUNSD License (non-commercial research and education; images under "
                   "third-party copyright)",
        "citation": "Jaume, Ekenel, Thiran, FUNSD: A Dataset for Form Understanding in Noisy "
                    "Scanned Documents, ICDAR-OST 2019",
    },
    "wildreceipt": {
        "dataset": "WildReceipt",
        "source": "https://mmocr.readthedocs.io/en/dev-1.x/user_guides/data_prepare/datasetzoo.html",
        "license": "none stated by the authors",
        "citation": "Sun et al., Spatial Dual-Modality Graph Reasoning for Key Information "
                    "Extraction, arXiv:2103.14470, 2021",
    },
    "sroie": {
        "dataset": "SROIE (ICDAR 2019), darentang/sroie pre-tagged lineage",
        "source": "https://rrc.cvc.uab.es/?ch=13",
        "license": "unclear (CC BY 4.0 asserted only by third-party mirrors)",
        "citation": "Huang et al., ICDAR2019 Competition on Scanned Receipt OCR and "
                    "Information Extraction, ICDAR 2019",
    },
}


def eligible(ds: str) -> pd.DataFrame:
    df = pd.read_parquet(REPO / "data_bundle" / ds / "processed" / "test.parquet")
    if ds != "cord":
        ev = set(json.loads((REPO / "data_bundle" / "splits" / f"{ds}_calib_seed42.json")
                            .read_text())["evaluation"])
        df = df[df["doc_id"].isin(ev)]
    df = df.assign(n_words=df["tokens"].map(len))
    return df.sort_values(["n_words", "doc_id"]).reset_index(drop=True)


def select(ds: str) -> list:
    df = eligible(ds)
    n = len(df)
    picks = [("representative", (n - 1) // 2)]
    if ds == "cord":
        picks.append(("diagnostic", int(np.floor(0.9 * (n - 1)))))
    return [(role, i, df.iloc[i], n) for role, i in picks]


def gold_spans(tags):
    out, start, field = [], None, None
    for i, t in enumerate(tags):
        if t.startswith("B-") or (t.startswith("I-") and t[2:] != field):
            if start is not None:
                out.append((start, i, field))
            start, field = i, t[2:]
        elif not t.startswith("I-"):
            if start is not None:
                out.append((start, i, field))
            start, field = None, None
    if start is not None:
        out.append((start, len(tags), field))
    return out


def ground_truth(predictor, probs, row):
    """Gold fields and whether each one is covered by the calibrated sets."""
    sets = predictor.calibrated_sets(probs)
    n = probs.shape[0]
    tags = list(row["ner_tags"])[:n]
    label2id = {v: k for k, v in predictor.id2label.items()}
    fields = []
    for a, b, f in gold_spans(tags):
        covered = bool(all(sets[i, label2id[tags[i]]] for i in range(a, b)))
        fields.append({"type": f, "text": " ".join(row["tokens"][a:b]), "word_span": [a, b],
                       "covered_by_calibrated_sets": covered})
    missed = sum(not f["covered_by_calibrated_sets"] for f in fields)
    return {"fields": fields, "n_gold_fields": len(fields), "n_missed": missed,
            "document_field_miss_fraction": missed / len(fields) if fields else 0.0}


def redact(result: dict) -> dict:
    for f in result["fields"]:
        f.pop("text"), f.pop("bbox")
    for c in result["unassigned_field_candidates"]:
        c.pop("text")
    result["redacted"] = ("words, boxes and the document image are omitted: the dataset's "
                          "redistribution terms are not cleared (docs/release/"
                          "DATASET_LICENSE_AUDIT.md). Re-run tools/showcase_examples.py "
                          "--full on your own copy of the dataset for the complete output.")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="MHamdan/",
                    help="prefix for calidoc-layoutlmv3-large-<ds> (hub namespace or local dir)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", default=str(REPO / "assets" / "examples"))
    ap.add_argument("--datasets", nargs="*", default=["sroie", "cord", "funsd", "wildreceipt"])
    ap.add_argument("--full", action="store_true", help="never redact (local use only)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    index = []
    for ds in args.datasets:
        model_id = f"{args.model_root}calidoc-layoutlmv3-large-{ds}"
        predictor = CaliDocPredictor.from_pretrained(model_id, device=args.device,
                                                     revision=args.revision)
        for role, rank, row, n_eligible in select(ds):
            image = Image.open(REPO / "data_bundle" / row["image_path"]).convert("RGB")
            words = [str(w) for w in row["tokens"]]
            boxes = [[int(v) for v in b] for b in row["bboxes"]]
            probs = predictor.word_probabilities(image, words, boxes)
            result = predictor.predict(image, words, boxes)
            cleared = ds in CLEARED or args.full
            example = {
                "dataset": ds,
                "doc_id": row["doc_id"],
                "split": "test",
                "role": role,
                "selection": {"rule": "sorted by (word count, doc_id) over eligible documents; "
                                      + ("lower median" if role == "representative"
                                         else "90th percentile"),
                              "rank": rank, "n_eligible": n_eligible,
                              "n_words": int(row["n_words"])},
                "attribution": ATTRIBUTION[ds],
                "ocr": "dataset-provided OCR words and boxes (no OCR engine is run)",
                "redistribution_cleared": ds in CLEARED,
            }
            if ds == "cord":
                example["attribution"]["source_index_in_split"] = int(row["source_image_id"])
            if cleared:
                example["ground_truth"] = ground_truth(predictor, probs, row)
                example["image"] = "document.png"
            result = result if cleared else redact(result)
            example["prediction"] = result
            out = Path(args.out) / ds / row["doc_id"]
            out.mkdir(parents=True, exist_ok=True)
            if cleared:
                image.save(out / "document.png")
                (out / "ocr.json").write_text(json.dumps(
                    {"words": words, "boxes": boxes, "box_format": "normalized_1000"},
                    ensure_ascii=False) + "\n")
            (out / "prediction.json").write_text(
                json.dumps(example, indent=2, ensure_ascii=False) + "\n")
            s = result["summary"]
            index.append({"dataset": ds, "doc_id": row["doc_id"], "role": role,
                          "model": model_id, "path": f"{ds}/{row['doc_id']}",
                          "n_fields": s["n_fields"], "status_counts": s["status_counts"],
                          "route": s["route"]})
            print(json.dumps(index[-1]), flush=True)
    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
