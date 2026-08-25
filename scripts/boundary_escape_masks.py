#!/usr/bin/env python3
"""Generalized positional audit: pre-specified masks beyond span edges.

Runs the conditional randomization test of `span_crc.boundary_escape_unit`
with the pre-specified position masks {edges, interior} on every dataset, and
controls multiplicity with Holm over the ENLARGED family of all (field, mask)
pairs -- the correction required as soon as more than one mask is tested.
Both masks are fixed a priori (they depend only on span length), so the
conditional randomization null remains exact for each; see
`span_crc.position_mask` for the validity conditions.

Purpose in the paper: a benign demonstration that the audit generalizes to
non-edge hypotheses. Expected outcome given the edge-concentrated SROIE
defect: the edge tests keep rejecting SROIE ADDRESS/COMPANY, the interior
tests reject nothing anywhere (interior escapes are *depleted* when edges are
enriched, and the test is one-sided for enrichment), and clean datasets stay
clean under both masks.

Usage:
    python3 scripts/boundary_escape_masks.py
Writes results/local_20260803/boundary_escape_masks.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_crc import (DATASETS, DEFAULT_CACHE, DEFAULT_RESULTS,  # noqa: E402
                      adjusted_pvalues, boundary_escape_unit, build_spans,
                      calibration_units, crc_lambda, holm_reject, load_cache,
                      _by_doc)

MASKS = ["edges", "interior"]
GAMMA = 0.10
N_BOOT = 5000
SEED = 0


def main() -> int:
    out = {"protocol": {"masks": MASKS, "gamma": GAMMA, "n_boot": N_BOOT,
                        "seed": SEED,
                        "family": "all (dataset-local field, mask) pairs, "
                                  "Holm per dataset",
                        "note": "masks are pre-specified functions of span "
                                "length only; the enlarged family corrects "
                                "for testing two masks"},
           "datasets": {}}
    for ds in DATASETS:
        z, meta, id2label = load_cache(ds, DEFAULT_CACHE)
        name, cal_key, _ = calibration_units(meta)[0]
        cal_p = z[f"{cal_key}_probs"]
        cal_spans, cal_lac = build_spans(cal_p, z[f"{cal_key}_labels"],
                                         z[f"{cal_key}_docstart"], id2label)
        lam0, _ = crc_lambda(_by_doc(cal_spans), 0.10)

        per_mask, pooled_p = {}, {}
        for mask in MASKS:
            r = boundary_escape_unit(cal_spans, cal_lac, cal_p, id2label,
                                     tau=lam0, n_boot=N_BOOT, seed=SEED,
                                     gamma=GAMMA, mask=mask)
            per_mask[mask] = r
            for f, rec in r["fields"].items():
                if rec["p_value"] is not None:
                    pooled_p[f"{f}::{mask}"] = rec["p_value"]

        dec = holm_reject(pooled_p, GAMMA)
        adj = adjusted_pvalues(pooled_p)
        rejected = sorted([k for k, v in dec.items() if v])
        out["datasets"][ds] = {
            "unit": name,
            "tau": float(lam0),
            "n_tests_in_family": len(pooled_p),
            "rejected_pairs_holm": rejected,
            "adjusted": {k: adj[k] for k in sorted(pooled_p)},
            "per_mask": {m: {
                "n_fields_tested": per_mask[m]["n_fields_tested"],
                "fields": {f: {kk: rec[kk] for kk in
                               ("n_informative_spans", "edge_observed",
                                "edge_expected", "edge_enrichment", "p_value")}
                           for f, rec in per_mask[m]["fields"].items()}
            } for m in MASKS},
        }
        print(f"{ds:<16} family={len(pooled_p):>2}  Holm rejections: "
              f"{rejected if rejected else 'none'}")

    dst = os.path.join(DEFAULT_RESULTS, "boundary_escape_masks.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
