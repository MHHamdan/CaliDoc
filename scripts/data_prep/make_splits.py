"""Generate deterministic protocol splits, persisted as JSON doc-ID lists.

Protocol constraints (structural fixes for the review-flagged violations):
- CORD: official 100-doc validation split ONCE (seed 42) into val_es (50,
  early stopping ONLY; temperature scaling is fit here) and cal (50, conformal
  calibration ONLY). Training code must never see cal.
- FUNSD: no official val. Three document-level 70/30 calibration/evaluation
  splits of the 50-doc test set (seeds 42, 123, 456), plus a 15-doc held-out
  early-stopping subset of train (seed 42) that must be removed from training.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import BUNDLE_DIR, hard_stop, status, write_json

import pyarrow.parquet as pq

FUNSD_CALIB_SEEDS = [42, 123, 456]
CORD_SPLIT_SEED = 42
FUNSD_ES_SEED = 42
FUNSD_ES_SIZE = 15


def doc_ids(dataset, split):
    table = pq.read_table(BUNDLE_DIR / dataset / "processed" / f"{split}.parquet",
                          columns=["doc_id"])
    return sorted(table.column("doc_id").to_pylist())


def shuffled(ids, seed):
    ids = list(ids)  # input is sorted -> deterministic given the seed
    random.Random(seed).shuffle(ids)
    return ids


def main():
    out = BUNDLE_DIR / "splits"
    out.mkdir(parents=True, exist_ok=True)
    ok = True

    # --- CORD: validation -> val_es (50) + cal (50), seed 42, disjoint ---
    val_ids = doc_ids("cord", "validation")
    if len(val_ids) != 100:
        hard_stop(f"CORD validation has {len(val_ids)} docs, expected 100")
    sh = shuffled(val_ids, CORD_SPLIT_SEED)
    val_es, cal = sorted(sh[:50]), sorted(sh[50:])
    if set(val_es) & set(cal):
        hard_stop("CORD val_es and cal overlap")
    write_json(out / "cord_val_es.json", {
        "dataset": "cord", "source_split": "validation", "seed": CORD_SPLIT_SEED,
        "purpose": "early stopping ONLY; temperature scaling is fit here",
        "doc_ids": val_es,
    })
    write_json(out / "cord_cal.json", {
        "dataset": "cord", "source_split": "validation", "seed": CORD_SPLIT_SEED,
        "purpose": "conformal calibration ONLY; training code must never see this",
        "doc_ids": cal,
    })
    ok &= status("cord", "split val_es/cal", "50/50 disjoint",
                 f"{len(val_es)}/{len(cal)}", len(val_es) == 50 and len(cal) == 50)

    # --- FUNSD: 3x 70/30 calib/eval splits of the 50-doc test set ---
    test_ids = doc_ids("funsd", "test")
    if len(test_ids) != 50:
        hard_stop(f"FUNSD test has {len(test_ids)} docs, expected 50")
    n_cal = round(0.7 * len(test_ids))  # 35
    for seed in FUNSD_CALIB_SEEDS:
        sh = shuffled(test_ids, seed)
        calib, evaln = sorted(sh[:n_cal]), sorted(sh[n_cal:])
        if set(calib) & set(evaln):
            hard_stop(f"FUNSD calib/eval overlap at seed {seed}")
        write_json(out / f"funsd_calib_seed{seed}.json", {
            "dataset": "funsd", "source_split": "test", "seed": seed,
            "purpose": "document-level 70/30 conformal calibration/evaluation split",
            "calibration": calib, "evaluation": evaln,
        })
        ok &= status("funsd", f"split calib seed{seed}", "35/15",
                     f"{len(calib)}/{len(evaln)}",
                     len(calib) == 35 and len(evaln) == 15)

    # --- FUNSD: 15-doc early-stopping holdout from train ---
    train_ids = doc_ids("funsd", "train")
    if len(train_ids) != 149:
        hard_stop(f"FUNSD train has {len(train_ids)} docs, expected 149")
    es = sorted(shuffled(train_ids, FUNSD_ES_SEED)[:FUNSD_ES_SIZE])
    write_json(out / "funsd_es.json", {
        "dataset": "funsd", "source_split": "train", "seed": FUNSD_ES_SEED,
        "purpose": "early stopping ONLY; these docs must be REMOVED from training",
        "doc_ids": es,
        "effective_train_size": len(train_ids) - len(es),
    })
    ok &= status("funsd", "split es holdout", 15, len(es), len(es) == 15)

    if not ok:
        hard_stop("split generation failed")


if __name__ == "__main__":
    main()
