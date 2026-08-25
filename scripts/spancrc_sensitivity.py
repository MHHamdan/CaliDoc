#!/usr/bin/env python3
"""Sensitivity companion to span_crc.py: candidate-label set sizes + m0 grid.

Two post-hoc analyses over the committed probability caches, no retraining:

1. **Candidate-label (Mondrian-by-candidate-field) set size.** The manuscript's
   field-conditional rule is defined per *candidate label*: the prediction set
   of token i is {y : 1 - p(y|x_i) <= lambda_{type(y)}}, where type(y) is the
   field of BIO label y and non-field labels (O) use the pooled lambda_0. Under
   this rule a gold type-t span is covered iff S(s) <= lambda_t -- identical to
   the coverage span_crc.py computes -- but the average set size differs from
   the gold-extent accounting span_crc.py used. This script computes the
   deployable candidate-label set size (the number the paper must report).

2. **m0 sensitivity.** The fallback support threshold min_cal_spans (m0) is
   swept over {10, 15, 25, 50}; for each value we report span coverage,
   realized per-document field-miss risk, span CovGap, candidate-label set
   size, and how many field types use the fallback. Descriptive only: m0 = 25
   remains the pre-specified operating point everywhere in the paper.

Writes results/<tag>/spancrc_sensitivity.json. Self-check: for the default
m0 = 25 the span coverage must reproduce span_crc_<ds>.json to 1e-6.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_crc import (DEFAULT_CACHE, DEFAULT_RESULTS, DATASETS, build_spans,
                      calibration_units, conformal_quantile, crc_lambda,
                      load_cache, _by_doc)

M0_GRID = [10, 15, 25, 50]


def field_lambdas(cal_spans, alpha, m0):
    lam0, _ = crc_lambda(_by_doc(cal_spans), alpha)
    groups = {}
    for s in cal_spans:
        groups.setdefault(s["field"], []).append(s)
    lam_f, n_fb = {}, 0
    for f, g in groups.items():
        if len(g) < m0:
            lam_f[f] = lam0
            n_fb += 1
        else:
            lam_f[f], _ = crc_lambda(_by_doc(g), alpha)
    return lam0, lam_f, n_fb, groups


def candidate_label_size(probs, id2label, lam0, lam_f):
    """Average set size under the candidate-label threshold rule."""
    n_labels = probs.shape[1]
    thr = np.full(n_labels, lam0)
    for j in range(n_labels):
        lab = id2label[j]
        if lab.startswith(("B-", "I-")):
            thr[j] = lam_f.get(lab[2:], lam0)
    return float((probs >= 1.0 - thr[None, :]).sum(1).mean())


def unit_eval(cal_spans, ev_spans, ev_probs, cal_lac, ev_lac, id2label, alpha, m0):
    lam0, lam_f, n_fb, groups = field_lambdas(cal_spans, alpha, m0)
    cov = float(np.mean([s["score"] <= lam_f.get(s["field"], lam0) for s in ev_spans]))
    # realized per-document field-miss risk (the quantity CRC controls)
    docs = {}
    for s in ev_spans:
        docs.setdefault(s["doc"], []).append(s["score"] > lam_f.get(s["field"], lam0))
    risk = float(np.mean([np.mean(v) for v in docs.values()]))
    fields = sorted({s["field"] for s in ev_spans})
    pf = {f: float(np.mean([s["score"] <= lam_f.get(f, lam0)
                            for s in ev_spans if s["field"] == f])) for f in fields}
    covgap = float(np.mean([abs(v - (1 - alpha)) for v in pf.values()]) * 100)
    return {
        "m0": m0,
        "span_coverage": cov,
        "ev_doc_risk": risk,
        "span_covgap_pp": covgap,
        "avg_set_size_candidate_rule": candidate_label_size(ev_probs, id2label, lam0, lam_f),
        "n_fields_fallback": n_fb,
        "n_fields_cal": len(groups),
        "qtok_size": float((ev_probs >= 1 - conformal_quantile(cal_lac, alpha)).sum(1).mean()),
    }


def main():
    alpha = 0.10
    out = {"alpha": alpha, "m0_grid": M0_GRID, "datasets": {}}
    for ds in DATASETS:
        z, meta, id2label = load_cache(ds, DEFAULT_CACHE)
        ref = json.load(open(os.path.join(DEFAULT_RESULTS, f"span_crc_{ds}.json")))
        per_m0 = {m0: [] for m0 in M0_GRID}
        for name, cal_key, ev_key in calibration_units(meta):
            cal_spans, cal_lac = build_spans(z[f"{cal_key}_probs"], z[f"{cal_key}_labels"],
                                             z[f"{cal_key}_docstart"], id2label)
            ev_spans, ev_lac = build_spans(z[f"{ev_key}_probs"], z[f"{ev_key}_labels"],
                                           z[f"{ev_key}_docstart"], id2label)
            for m0 in M0_GRID:
                per_m0[m0].append(unit_eval(cal_spans, ev_spans, z[f"{ev_key}_probs"],
                                            cal_lac, ev_lac, id2label, alpha, m0))
        agg = {}
        for m0, units in per_m0.items():
            agg[str(m0)] = {k: {"mean": float(np.mean([u[k] for u in units])),
                                "std": float(np.std([u[k] for u in units]))}
                            for k in units[0] if k != "m0"}
        # self-check against the committed artifact at the default m0
        got = agg["25"]["span_coverage"]["mean"]
        want = ref["aggregate"]["span_cov_field_crc"]["mean"]
        assert abs(got - want) < 1e-6, f"{ds}: {got} != committed {want}"
        out["datasets"][ds] = agg
        print(f"{ds}: self-check PASS (m0=25 coverage {got:.4f} == committed)")
        for m0 in M0_GRID:
            a = agg[str(m0)]
            print(f"  m0={m0:>2}  cov={a['span_coverage']['mean']:.4f}"
                  f"  risk={a['ev_doc_risk']['mean']:.4f}"
                  f"  covgap={a['span_covgap_pp']['mean']:.2f}pp"
                  f"  size(cand)={a['avg_set_size_candidate_rule']['mean']:.3f}"
                  f"  size(tok)={a['qtok_size']['mean']:.3f}"
                  f"  fallback={a['n_fields_fallback']['mean']:.0f}/{a['n_fields_cal']['mean']:.0f}")
    dst = os.path.join(DEFAULT_RESULTS, "spancrc_sensitivity.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
