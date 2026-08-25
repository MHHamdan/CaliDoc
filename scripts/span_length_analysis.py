#!/usr/bin/env python3
"""Span-length stratification, aggregation-score diagnostics, field-support
economics and operational set-size statistics for SpanCRC.

Answers four reviewer questions from the cached probabilities alone (no GPU,
no retraining):

1. **Does max-score conservatism grow with span length?** (review Q1/Q4)
   Length-stratified full-span coverage, field-miss rate, mean span score and
   candidate-label set size, plus Spearman correlations between span length
   and (a) the max-token score S(s), (b) the miss indicator, (c) boundary-
   audit eligibility. Length buckets are fixed a priori and dyadic --
   {1}, {2}, {3-4}, {5-8}, {>=9} -- chosen before inspecting stratified
   results, to avoid favourable-bin selection.

2. **Are alternative span aggregations viable?** (review Q1)
   Mean, trimmed max (second-largest token score), and the within-span 0.9-
   quantile are *diagnostic ablations only*. Each alternative A defines a
   different reliability event {A(s) <= lambda}: calibrating CRC on A controls
   that event's risk, NOT complete-field coverage. Full-span coverage occurs
   iff every token is covered iff S_max(s) <= lambda, so the max is the unique
   aggregation whose sub-level event equals the delivered-field event. For
   each alternative we therefore report BOTH its own-event rate (which lands
   near 1-alpha, as its theorem promises) and the true full-span coverage and
   realized document-level field-miss risk it induces (which it does not
   control) -- stratified by length, where the gap must widen.

3. **Why is CORD expensive?** (field-support economics)
   Per-field: calibration support, threshold source (own lambda_t vs pooled
   fallback), lambda_t, coverage, and the field's candidate-admission rate --
   the fraction of ALL evaluation tokens whose prediction set admits at least
   one label of that field under the candidate-label rule. Summaries: fields
   above/below m0, share of evaluation spans belonging to fallback fields.

4. **What does a large set mean operationally?**
   Token-level set-size distribution (empty / singleton / multi rates, mean,
   p90, p95) and span-level review statistics (spans fully singleton, spans
   containing any multi-label token, spans containing any empty set) under
   both the token-marginal threshold and deployable field-CRC candidate rule.

Everything reuses the span construction and CRC primitives of
`scripts/span_crc.py` verbatim, runs per calibration unit and reports the
mean over units exactly as `span_crc.py` does.

Usage:
    python3 scripts/span_length_analysis.py --all
    python3 scripts/span_length_analysis.py --dataset cord
Writes results/local_20260803/span_length_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_crc import (DATASETS, DEFAULT_CACHE, DEFAULT_RESULTS, CACHE_TAG,  # noqa: E402
                      build_spans, calibration_units, conformal_quantile,
                      crc_lambda, load_cache, _by_doc)

# Fixed a-priori dyadic buckets; the audit's structural blind spot is L <= 2,
# which buckets 1 and 2 isolate exactly.
BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10**9)]
BUCKET_NAMES = ["1", "2", "3-4", "5-8", ">=9"]


def bucket_of(L: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= L <= hi:
            return i
    raise ValueError(L)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties (numpy only)."""
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average tied ranks
        vs = v[order]
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[j + 1] == vs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


# ------------------------------------------------------------------ scores

def span_token_scores(span, lac):
    return lac[span["lo"]:span["hi"]]


def agg_score(span, lac, kind: str) -> float:
    seg = span_token_scores(span, lac)
    if kind == "max":
        return float(seg.max())
    if kind == "mean":
        return float(seg.mean())
    if kind == "trimmed_max":            # second-largest; = max when L == 1
        return float(np.sort(seg)[-2]) if len(seg) >= 2 else float(seg.max())
    if kind == "q90":                    # within-span 0.9-quantile
        return float(np.quantile(seg, 0.9))
    raise ValueError(kind)


AGGS = ["max", "mean", "trimmed_max", "q90"]


def scores_by_doc(spans, lac, kind):
    d: Dict[int, List[float]] = {}
    for s in spans:
        d.setdefault(s["doc"], []).append(agg_score(s, lac, kind))
    return [np.asarray(v) for v in d.values()]


# ------------------------------------------------------------------ pieces

def length_stats(spans) -> dict:
    L = np.array([s["len"] for s in spans])
    per_field = {}
    for f in sorted({s["field"] for s in spans}):
        lf = np.array([s["len"] for s in spans if s["field"] == f])
        per_field[f] = {"n": int(len(lf)), "median": float(np.median(lf)),
                        "mean": float(lf.mean()), "p90": float(np.quantile(lf, 0.9)),
                        "max": int(lf.max())}
    return {
        "n_spans": int(len(L)),
        "median": float(np.median(L)), "mean": float(L.mean()),
        "p90": float(np.quantile(L, 0.9)), "max": int(L.max()),
        "bucket_counts": {BUCKET_NAMES[i]: int(sum(bucket_of(l) == i for l in L))
                          for i in range(len(BUCKETS))},
        "per_field": per_field,
    }


def stratified(ev_spans, ev_lac, ev_probs, lam_of, thr_vec) -> dict:
    """Coverage / miss / score / set size by length bucket for one rule.

    lam_of(span) -> threshold applied to that span's max score.
    thr_vec: per-label admission thresholds of the same rule (for set sizes).
    """
    out = {}
    sizes_tok = (ev_probs >= 1.0 - thr_vec[None, :]).sum(1)
    for i, name in enumerate(BUCKET_NAMES):
        grp = [s for s in ev_spans if bucket_of(s["len"]) == i]
        if not grp:
            out[name] = None
            continue
        cov = float(np.mean([s["score"] <= lam_of(s) for s in grp]))
        tok_idx = np.concatenate([np.arange(s["lo"], s["hi"]) for s in grp])
        out[name] = {
            "n_spans": len(grp),
            "full_span_coverage": cov,
            "field_miss_rate": 1.0 - cov,
            "mean_max_score": float(np.mean([s["score"] for s in grp])),
            "avg_token_set_size": float(sizes_tok[tok_idx].mean()),
        }
    return out


def eligibility_by_length(cal_spans, cal_lac, tau) -> dict:
    out = {}
    L_all, elig_all = [], []
    for i, name in enumerate(BUCKET_NAMES):
        grp = [s for s in cal_spans if bucket_of(s["len"]) == i]
        if not grp:
            out[name] = None
            continue
        flags = []
        for s in grp:
            k = int((span_token_scores(s, cal_lac) > tau).sum())
            flags.append(0 < k < s["len"])
            L_all.append(s["len"])
            elig_all.append(flags[-1])
        out[name] = {"n_spans": len(grp),
                     "informative_rate": float(np.mean(flags))}
    return {"by_bucket": out,
            "spearman_length_vs_eligibility":
                spearman(np.array(L_all), np.array(elig_all, dtype=float))}


def set_size_stats(sizes: np.ndarray) -> dict:
    return {
        "mean": float(sizes.mean()),
        "p50": float(np.quantile(sizes, 0.50)),
        "p90": float(np.quantile(sizes, 0.90)),
        "p95": float(np.quantile(sizes, 0.95)),
        "empty_rate": float((sizes == 0).mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "multi_rate": float((sizes >= 2).mean()),
    }


def span_review_stats(ev_spans, sizes: np.ndarray) -> dict:
    all_single, any_multi, any_empty = [], [], []
    for s in ev_spans:
        seg = sizes[s["lo"]:s["hi"]]
        all_single.append(bool((seg == 1).all()))
        any_multi.append(bool((seg >= 2).any()))
        any_empty.append(bool((seg == 0).any()))
    return {
        "span_all_singleton_rate": float(np.mean(all_single)),
        "span_any_multi_rate": float(np.mean(any_multi)),
        "span_any_empty_rate": float(np.mean(any_empty)),
    }


# ------------------------------------------------------------------ unit run

def run_unit(cal_spans, ev_spans, cal_lac, ev_lac, ev_probs, id2label,
             alpha, m0) -> dict:
    n_labels = ev_probs.shape[1]

    # --- thresholds identical to span_crc.py -----------------------------
    qtok = conformal_quantile(cal_lac, alpha)
    lam0, _ = crc_lambda(_by_doc(cal_spans), alpha)
    cal_fields: Dict[str, list] = {}
    for s in cal_spans:
        cal_fields.setdefault(s["field"], []).append(s)
    lam_f = {}
    for f, grp in cal_fields.items():
        lam_f[f] = lam0 if len(grp) < m0 else crc_lambda(_by_doc(grp), alpha)[0]

    def thr_vec(rule: str) -> np.ndarray:
        t = np.full(n_labels, {"tok": qtok, "pool": lam0}.get(rule, lam0))
        if rule == "field":
            for j in range(n_labels):
                lab = id2label[j]
                t[j] = lam_f.get(lab[2:], lam0) if lab.startswith(("B-", "I-")) else lam0
        return t

    rules = {
        "token_marginal": (lambda s: qtok, thr_vec("tok")),
        "pooled_crc": (lambda s: lam0, thr_vec("pool")),
        "field_crc": (lambda s: lam_f.get(s["field"], lam0), thr_vec("field")),
    }

    # --- 1. length stratification ---------------------------------------
    L = np.array([s["len"] for s in ev_spans], dtype=float)
    S = np.array([s["score"] for s in ev_spans], dtype=float)
    miss_fc = np.array([s["score"] > lam_f.get(s["field"], lam0)
                        for s in ev_spans], dtype=float)
    strat = {name: stratified(ev_spans, ev_lac, ev_probs, lam, tv)
             for name, (lam, tv) in rules.items()}
    correlations = {
        "spearman_length_vs_max_score": spearman(L, S),
        "spearman_length_vs_miss_field_crc": spearman(L, miss_fc),
    }

    # --- 2. aggregation ablation (diagnostic only) -----------------------
    agg_out = {}
    ev_docs_max = _by_doc(ev_spans)                      # true-event doc risks
    for kind in AGGS:
        lamA, _ = crc_lambda(scores_by_doc(cal_spans, cal_lac, kind), alpha)
        ownA = np.array([agg_score(s, ev_lac, kind) for s in ev_spans])
        own_rate = float((ownA <= lamA).mean())
        true_cov = float((S <= lamA).mean())
        true_risk = float(np.mean([(d > lamA).mean() for d in ev_docs_max]))
        size = float((ev_probs >= 1.0 - lamA).sum(1).mean())
        by_bucket = {}
        for i, name in enumerate(BUCKET_NAMES):
            m = np.array([bucket_of(s["len"]) == i for s in ev_spans])
            by_bucket[name] = ({"n_spans": int(m.sum()),
                                "true_full_span_coverage": float((S[m] <= lamA).mean()),
                                "own_event_rate": float((ownA[m] <= lamA).mean())}
                               if m.any() else None)
        agg_out[kind] = {
            "lambda": float(lamA),
            "own_event_rate": own_rate,
            "true_full_span_coverage": true_cov,
            "true_doc_field_miss_risk": true_risk,
            "avg_token_set_size": size,
            "controls_full_span_event": kind == "max",
            "by_bucket": by_bucket,
        }

    # --- 3. field-support economics --------------------------------------
    admit_field = {}
    for f, lam in lam_f.items():
        cols = [j for j in range(n_labels)
                if id2label[j].startswith(("B-", "I-")) and id2label[j][2:] == f]
        if cols:
            admit_field[f] = float(
                (ev_probs[:, cols] >= 1.0 - lam).any(1).mean())
    per_field = {}
    for f in sorted({s["field"] for s in ev_spans} | set(cal_fields)):
        fe = [s for s in ev_spans if s["field"] == f]
        n_cal = len(cal_fields.get(f, []))
        per_field[f] = {
            "n_cal_spans": n_cal,
            "n_ev_spans": len(fe),
            "fallback": n_cal < m0,
            "lambda": float(lam_f.get(f, lam0)),
            "span_cov_field_crc": (float(np.mean([s["score"] <= lam_f.get(f, lam0)
                                                  for s in fe])) if fe else None),
            "candidate_admission_rate": admit_field.get(f),
            "median_len_ev": float(np.median([s["len"] for s in fe])) if fe else None,
        }
    n_fb = sum(1 for v in per_field.values() if v["fallback"])
    ev_in_fb = float(np.mean([per_field[s["field"]]["fallback"] for s in ev_spans]))
    support = {
        "m0": m0,
        "n_field_types": len(per_field),
        "n_fields_fallback": n_fb,
        "n_fields_own_threshold": len(per_field) - n_fb,
        "frac_ev_spans_in_fallback_fields": ev_in_fb,
        "per_field": per_field,
    }

    # --- 4. operational set-size statistics ------------------------------
    operational = {}
    for name, (_, tv) in rules.items():
        sizes = (ev_probs >= 1.0 - tv[None, :]).sum(1)
        operational[name] = {**set_size_stats(sizes),
                             **span_review_stats(ev_spans, sizes)}

    # --- eligibility vs length (calibration side, where the test runs) ----
    elig = eligibility_by_length(cal_spans, cal_lac, lam0)

    return {
        "thresholds": {"qhat_token": float(qtok), "lambda_pooled": float(lam0)},
        "length_stats_ev": length_stats(ev_spans),
        "length_stats_cal": length_stats(cal_spans),
        "stratified_reliability": strat,
        "correlations": correlations,
        "aggregation_ablation": agg_out,
        "field_support": support,
        "operational_set_size": operational,
        "boundary_audit_eligibility_by_length": elig,
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
        res = run_unit(cal_spans, ev_spans, cal_lac, ev_lac, ev_p, id2label,
                       alpha, m0)
        res["unit"] = name
        per_unit.append(res)
    return {
        "dataset": dataset,
        "protocol": {
            "alpha": alpha, "m0": m0,
            "buckets": BUCKET_NAMES,
            "bucket_definition": "fixed a-priori dyadic buckets",
            "aggregations": AGGS,
            "note": ("Alternative aggregations are DIAGNOSTIC: each controls "
                     "risk of its own event {A(s)<=lambda}, not complete-field "
                     "coverage; only the max score's sub-level event equals "
                     "the full-span-coverage event."),
            "cache": f"results/cache/{dataset}_{CACHE_TAG}.npz",
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
        print(f"\n=== {ds.upper()} ===")
        ls = u0["length_stats_ev"]
        print(f"  span length: median {ls['median']:.0f}, mean {ls['mean']:.2f}, "
              f"p90 {ls['p90']:.0f}, max {ls['max']}")
        print(f"  Spearman(L, S_max) = {u0['correlations']['spearman_length_vs_max_score']:.3f}")
        for k, v in u0["aggregation_ablation"].items():
            print(f"  agg {k:<12} own-event {v['own_event_rate']:.3f}  "
                  f"true full-span {v['true_full_span_coverage']:.3f}  "
                  f"true doc risk {v['true_doc_field_miss_risk']:.3f}")
        fs = u0["field_support"]
        print(f"  fields: {fs['n_fields_own_threshold']} own / "
              f"{fs['n_fields_fallback']} fallback; "
              f"{fs['frac_ev_spans_in_fallback_fields']*100:.1f}% of ev spans in fallback fields")
        op = u0["operational_set_size"]["field_crc"]
        print(f"  field-CRC sets: mean {op['mean']:.2f}, p90 {op['p90']:.0f}, "
              f"p95 {op['p95']:.0f}, singleton {op['singleton_rate']*100:.1f}%, "
              f"multi {op['multi_rate']*100:.1f}%, empty {op['empty_rate']*100:.1f}%")

    dst = os.path.join(args.results_dir, "span_length_analysis.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
