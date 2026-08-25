#!/usr/bin/env python3
"""Support-aware interpolation of field-conditional SpanCRC thresholds.

Ports the interpolated-quantile idea of Ding, Fermanian & Salmon,
"Conformal Prediction for Long-Tailed Classification" (ICLR 2026,
arXiv:2507.06867) -- their INTERP-Q rule interpolates the classwise
(Mondrian) quantile with the marginal quantile,

    q_y(tau) = tau * q_y^classwise + (1 - tau) * q^marginal,

with tau in [0,1] a user-chosen knob and degenerate classwise quantiles
clipped to 1 -- from token-level coverage to field-conditional CRC thresholds:

    lambda_t(tau) = tau * lambda_t^raw + (1 - tau) * lambda_0,

where lambda_0 is the pooled document-level CRC threshold and lambda_t^raw is
the per-field CRC threshold computed WITHOUT any m0 fallback (degenerate
low-support solutions return 1.0, the analogue of INTERP-Q's clipping).

Honest scope, stated in the artifact and the paper: interpolated thresholds
inherit NO field-conditional finite-sample guarantee. INTERP-Q's proven
guarantee in the token-coverage setting is only marginal coverage >= 1-2alpha;
no analogous CRC statement is claimed here. tau is swept on a fixed grid that
is reported in full; no value is selected on the final evaluation set.

Comparison set on each dataset/unit:
  * pooled CRC                       (= tau = 0)
  * current field-CRC + m0 fallback  (the paper's deployed method)
  * interpolation grid tau in {0.25, 0.5, 0.75}
  * full per-field CRC, no fallback  (= tau = 1; rare fields degenerate to 1)

Metrics per rule: realized per-document field-miss risk, full-span coverage,
span CovGap, average / p90 set size, empty-set rate, and the same metrics
restricted to supported (n_cal >= m0) and fallback (n_cal < m0) field strata.

Usage:
    python3 scripts/spancrc_interpolation.py --dataset cord
    python3 scripts/spancrc_interpolation.py --all
Writes results/local_20260803/spancrc_interpolation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_crc import (DEFAULT_CACHE, DEFAULT_RESULTS, CACHE_TAG,  # noqa: E402
                      build_spans, calibration_units, crc_lambda, load_cache,
                      _by_doc)

DATASETS = ["cord", "funsd", "sroie", "wildreceipt", "sroie_repaired"]
TAUS = [0.25, 0.50, 0.75]


def evaluate_rule(name, lam_of_field, ev_spans, ev_probs, id2label, lam_default,
                  alpha, support) -> dict:
    """Metrics for a per-field threshold map under the candidate-label rule."""
    n_labels = ev_probs.shape[1]
    thr = np.full(n_labels, lam_default, dtype=float)
    for j in range(n_labels):
        lab = id2label[j]
        if lab.startswith(("B-", "I-")):
            thr[j] = lam_of_field.get(lab[2:], lam_default)
    sizes = (ev_probs >= 1.0 - thr[None, :]).sum(1)

    def span_metrics(spans):
        if not spans:
            return None
        cov = [s["score"] <= lam_of_field.get(s["field"], lam_default)
               for s in spans]
        docs: Dict[int, List[bool]] = {}
        for s, c in zip(spans, cov):
            docs.setdefault(s["doc"], []).append(not c)
        per_field_cov: Dict[str, List[bool]] = {}
        for s, c in zip(spans, cov):
            per_field_cov.setdefault(s["field"], []).append(c)
        gaps = [abs(float(np.mean(v)) - (1 - alpha)) * 100
                for v in per_field_cov.values()]
        return {
            "n_spans": len(spans),
            "full_span_coverage": float(np.mean(cov)),
            "doc_field_miss_risk": float(np.mean([np.mean(v) for v in docs.values()])),
            "span_covgap_pp": float(np.mean(gaps)),
        }

    supported = [s for s in ev_spans if support.get(s["field"], 0) >= 0
                 and not support["_fallback"][s["field"]]]
    fallback = [s for s in ev_spans if support["_fallback"][s["field"]]]
    return {
        "rule": name,
        "overall": span_metrics(ev_spans),
        "supported_fields": span_metrics(supported),
        "fallback_fields": span_metrics(fallback),
        "avg_set_size": float(sizes.mean()),
        "p90_set_size": float(np.quantile(sizes, 0.90)),
        "empty_set_rate": float((sizes == 0).mean()),
    }


def run_dataset(dataset, cache_dir, alpha, m0) -> dict:
    z, meta, id2label = load_cache(dataset, cache_dir)
    per_unit = []
    for name, cal_key, ev_key in calibration_units(meta):
        cal_p, cal_y, cal_ds = (z[f"{cal_key}_probs"], z[f"{cal_key}_labels"],
                                z[f"{cal_key}_docstart"])
        ev_p, ev_y, ev_ds = (z[f"{ev_key}_probs"], z[f"{ev_key}_labels"],
                             z[f"{ev_key}_docstart"])
        cal_spans, cal_lac = build_spans(cal_p, cal_y, cal_ds, id2label)
        ev_spans, ev_lac = build_spans(ev_p, ev_y, ev_ds, id2label)
        if not cal_spans or not ev_spans:
            continue

        lam0, _ = crc_lambda(_by_doc(cal_spans), alpha)
        cal_fields: Dict[str, list] = {}
        for s in cal_spans:
            cal_fields.setdefault(s["field"], []).append(s)

        # raw per-field CRC with NO fallback; degenerate solutions are 1.0,
        # the CRC analogue of INTERP-Q's clipped classwise quantile.
        lam_raw = {f: crc_lambda(_by_doc(g), alpha)[0]
                   for f, g in cal_fields.items()}
        # the paper's deployed rule: m0 fallback to pooled.
        lam_m0 = {f: (lam0 if len(g) < m0 else lam_raw[f])
                  for f, g in cal_fields.items()}
        support = {f: len(g) for f, g in cal_fields.items()}
        support["_fallback"] = {f: len(cal_fields.get(f, [])) < m0
                                for f in set(list(cal_fields) +
                                             [s["field"] for s in ev_spans])}

        rules = [("pooled_crc", {f: lam0 for f in lam_raw}),
                 ("field_crc_m0_fallback", lam_m0)]
        for tau in TAUS:
            rules.append((f"interp_tau_{tau:.2f}",
                          {f: tau * lam_raw[f] + (1 - tau) * lam0
                           for f in lam_raw}))
        rules.append(("field_crc_no_fallback", lam_raw))

        # ---- level-calibrated interpolation (formally guaranteed) ---------
        # Shape lam_t^A from calibration half A; interpolation LEVEL mu
        # chosen by CRC over half B's pooled per-document miss fraction:
        #   thr_t(mu) = min(1, tau*lam_t^A + (1-tau)*mu).
        # Conditional on A, L_j(mu) is non-increasing, right-continuous,
        # [0,1]-bounded with L_j(1/(1-tau)) = 0 (every threshold reaches 1),
        # so the CRC argument applies verbatim over B's documents and yields
        # E[R(thr(mu_hat))] <= alpha for a fresh document -- the same pooled
        # document-level bound as pooled CRC, delivered by a support-aware
        # field-shaped profile. Split is pre-specified: documents sorted by
        # index, first half -> A (shape), second half -> B (level).
        doc_ids = sorted({s["doc"] for s in cal_spans})
        half = (len(doc_ids) + 1) // 2
        docs_A, docs_B = set(doc_ids[:half]), set(doc_ids[half:])
        spans_A = [s for s in cal_spans if s["doc"] in docs_A]
        spans_B = [s for s in cal_spans if s["doc"] in docs_B]
        fields_A: Dict[str, list] = {}
        for s in spans_A:
            fields_A.setdefault(s["field"], []).append(s)
        lam_shape = {f: crc_lambda(_by_doc(g), alpha)[0]
                     for f, g in fields_A.items()}
        # shape default for fields absent from A must be a function of A
        # ONLY (Proposition 3 hypothesis): pooled CRC over half A.
        lam0_A = crc_lambda(_by_doc(spans_A), alpha)[0]

        def crc_level(tau: float) -> float:
            """CRC over B's documents for the interpolation level mu."""
            mu_max = 1.0 / (1.0 - tau)
            cands = set([0.0, mu_max])
            for s in spans_B:
                lam_a = lam_shape.get(s["field"], lam0_A)
                mu = (s["score"] - tau * lam_a) / (1.0 - tau)
                if 0.0 <= mu <= mu_max:
                    cands.add(mu)
            by_doc: Dict[int, list] = {}
            for s in spans_B:
                by_doc.setdefault(s["doc"], []).append(s)
            n = len(by_doc)
            for mu in sorted(cands):
                risk = float(np.mean([
                    np.mean([s["score"] > min(1.0, tau * lam_shape.get(s["field"], lam0_A)
                                              + (1 - tau) * mu) for s in grp])
                    for grp in by_doc.values()]))
                if (n * risk + 1.0) / (n + 1) <= alpha:
                    return mu
            return mu_max

        for tau in TAUS:
            mu_hat = crc_level(tau)
            # profile defined over the full label schema (cal AND eval field
            # names -- schema only, no evaluation scores) so unseen fields
            # get the calibrated profile, not the raw pooled default.
            all_fields = set(list(lam_raw) + list(lam_shape)
                             + [s["field"] for s in ev_spans])
            rules.append((f"crc_interp_tau_{tau:.2f}",
                          {f: min(1.0, tau * lam_shape.get(f, lam0_A)
                                  + (1 - tau) * mu_hat)
                           for f in all_fields}))

        res = {"unit": name,
               "lambda_pooled": float(lam0),
               "n_fields": len(cal_fields),
               "n_fields_fallback": int(sum(1 for f in cal_fields
                                            if len(cal_fields[f]) < m0)),
               "rules": [evaluate_rule(nm, lam, ev_spans, ev_p, id2label,
                                       lam0, alpha, support)
                         for nm, lam in rules]}
        per_unit.append(res)

    return {
        "dataset": dataset,
        "protocol": {
            "alpha": alpha, "m0": m0, "tau_grid": TAUS,
            "cache": f"results/cache/{dataset}_{CACHE_TAG}.npz",
            "method": ("lambda_t(tau) = tau*lambda_t^raw + (1-tau)*lambda_0; "
                       "degenerate per-field CRC solutions clipped to 1.0. "
                       "Port of INTERP-Q (Ding, Fermanian & Salmon, ICLR 2026) "
                       "to field-conditional CRC thresholds."),
            "guarantee": ("NONE at the field-conditional level. Interpolated "
                          "thresholds are heuristics; INTERP-Q's own proven "
                          "guarantee in the coverage setting is marginal "
                          ">= 1-2alpha. The tau grid is reported in full and "
                          "never selected on the evaluation set."),
        },
        "per_unit": per_unit,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE)
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--m0", type=int, default=25)
    args = ap.parse_args()
    targets = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not targets:
        ap.error("pass --dataset or --all")

    out = {"alpha": args.alpha, "m0": args.m0, "datasets": {}}
    for ds in targets:
        r = run_dataset(ds, args.cache_dir, args.alpha, args.m0)
        out["datasets"][ds] = r
        u0 = r["per_unit"][0]
        print(f"\n=== {ds.upper()} (unit {u0['unit']}; "
              f"{u0['n_fields_fallback']}/{u0['n_fields']} fields below m0) ===")
        hdr = f"{'rule':<24}{'cov':>7}{'risk':>7}{'gap':>7}{'size':>7}{'p90':>6}{'empty':>7}"
        print(hdr)
        for rl in u0["rules"]:
            o = rl["overall"]
            print(f"{rl['rule']:<24}{o['full_span_coverage']:>7.3f}"
                  f"{o['doc_field_miss_risk']:>7.3f}{o['span_covgap_pp']:>7.1f}"
                  f"{rl['avg_set_size']:>7.2f}{rl['p90_set_size']:>6.0f}"
                  f"{rl['empty_set_rate']*100:>6.1f}%")

    dst = os.path.join(args.results_dir, "spancrc_interpolation.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
