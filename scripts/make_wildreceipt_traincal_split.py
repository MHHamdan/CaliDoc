#!/usr/bin/env python3
"""Generate the persisted clean-calibration split for wildreceipt_traincal.

Pre-designates 100 conformal-calibration documents from WildReceipt's official
training partition, disjoint from the existing 100-doc val_es carve. These
documents are excluded from fitting (train.py's stage='fit' guard enforces
this via calibration_doc_ids), so the retrained checkpoint never sees them and
the official 472-doc test set is exclusively final evaluation.

Deterministic: doc ids sorted, then numpy default_rng(seed) choice.
Refuses to overwrite an existing split file (the split is a protocol artifact).

Usage: python3 scripts/make_wildreceipt_traincal_split.py [--seed 42] [--n 100]
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "data_bundle"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    out = BUNDLE / "splits" / "wildreceipt_traincal_cal.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite protocol artifact {out}")

    train_ids = set(pd.read_parquet(
        BUNDLE / "wildreceipt" / "processed" / "train.parquet",
        columns=["doc_id"])["doc_id"])
    val_es = set(json.load(open(
        BUNDLE / "splits" / "wildreceipt_val_es.json"))["doc_ids"])
    assert len(train_ids) == 1267 and val_es <= train_ids and len(val_es) == 100

    pool = sorted(train_ids - val_es)          # 1,167 candidates
    rng = np.random.default_rng(args.seed)
    cal = sorted(rng.choice(pool, size=args.n, replace=False).tolist())

    out.write_text(json.dumps({
        "dataset": "wildreceipt_traincal",
        "purpose": ("conformal calibration ONLY; excluded from training, "
                    "early stopping and temperature fitting. Official test "
                    "set remains untouched final evaluation."),
        "created": str(date.today()),
        "seed": args.seed,
        "n_docs": len(cal),
        "source": "official train partition minus wildreceipt_val_es",
        "doc_ids": cal,
    }, indent=2))
    print(f"wrote {out}: {len(cal)} docs (train remains "
          f"{len(pool) - len(cal)}, val_es 100, test 472)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
