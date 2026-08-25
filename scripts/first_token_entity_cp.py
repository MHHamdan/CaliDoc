#!/usr/bin/env python3
"""First-token entity-level CP on WildReceipt at alpha = 0.02.

Mirrors the protocol of Rombach & Mehdiyev (IJDAR 2026) as closely as our
artifacts allow, for a qualified comparison: the prediction unit is the first
token of each gold entity span, the guarantee is marginal over entity
instances, and alpha = 0.02. Differences that remain and must be stated
wherever these numbers are used: they score entities by the negated raw gold
logit of a LayoutLMv2-Large under a 23-class segment-level schema with one
fixed 100-receipt calibration split; we use temperature-scaled LAC scores of a
LayoutLMv3-Large under the 49-label word-level BIO schema with document-level
70/30 calibration/evaluation splits over three seeds.

For the same threshold we also report the fraction of *complete* gold spans
whose tokens are all covered -- the quantity the first-token unit does not
certify -- and the SpanCRC pooled result at the same alpha for contrast.

Writes results/<tag>/first_token_entity_cp.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_crc import (DEFAULT_CACHE, DEFAULT_RESULTS, build_spans,
                      calibration_units, conformal_quantile, crc_lambda,
                      load_cache, _by_doc)

RUNS = [("wildreceipt", 0.02), ("sroie", 0.10), ("sroie", 0.02)]


def run(ds, ALPHA):
    z, meta, id2label = load_cache(ds, DEFAULT_CACHE)
    units = calibration_units(meta)
    rows = []
    for name, cal_key, ev_key in units:
        cal_p, cal_y, cal_ds = z[f"{cal_key}_probs"], z[f"{cal_key}_labels"], z[f"{cal_key}_docstart"]
        ev_p, ev_y, ev_ds = z[f"{ev_key}_probs"], z[f"{ev_key}_labels"], z[f"{ev_key}_docstart"]
        cal_spans, cal_lac = build_spans(cal_p, cal_y, cal_ds, id2label)
        ev_spans, ev_lac = build_spans(ev_p, ev_y, ev_ds, id2label)

        # first-token entity CP: calibrate on first-token LAC scores
        cal_ft = np.array([cal_lac[s["lo"]] for s in cal_spans])
        q = conformal_quantile(cal_ft, ALPHA)
        ev_ft = np.array([ev_lac[s["lo"]] for s in ev_spans])
        ft_cov = float((ev_ft <= q).mean())
        ft_idx = np.array([s["lo"] for s in ev_spans])
        ft_size = float((ev_p[ft_idx] >= 1 - q).sum(1).mean())
        ft_single = float(((ev_p[ft_idx] >= 1 - q).sum(1) == 1).mean())
        # full-span coverage at the same threshold
        full_cov = float(np.mean([s["score"] <= q for s in ev_spans]))
        # SpanCRC pooled at the same alpha
        lam0, _ = crc_lambda(_by_doc(cal_spans), ALPHA)
        span_cov_crc = float(np.mean([s["score"] <= lam0 for s in ev_spans]))
        size_crc = float((ev_p >= 1 - lam0).sum(1).mean())
        rows.append({"unit": name, "n_cal_entities": len(cal_spans),
                     "n_ev_entities": len(ev_spans), "qhat": float(q),
                     "first_token_coverage": ft_cov,
                     "first_token_avg_set_size": ft_size,
                     "first_token_singleton_rate": ft_single,
                     "full_span_coverage_same_threshold": full_cov,
                     "spancrc_pooled_lambda": float(lam0),
                     "spancrc_span_coverage": span_cov_crc,
                     "spancrc_avg_set_size": size_crc})

    def agg(k):
        v = [r[k] for r in rows]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    return {"dataset": ds, "alpha": ALPHA,
            "aggregate": {k: agg(k) for k in rows[0] if k != "unit"},
            "per_unit": rows}


def main():
    out = {"protocol_note": __doc__.strip().splitlines()[0], "runs": []}
    for ds, alpha in RUNS:
        r = run(ds, alpha)
        out["runs"].append(r)
        print(f"== {ds} alpha={alpha}")
        for k, v in r["aggregate"].items():
            print(f"  {k:38s} {v['mean']:.4f} +- {v['std']:.4f}")
    dst = os.path.join(DEFAULT_RESULTS, "first_token_entity_cp.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
