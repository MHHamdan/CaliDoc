#!/usr/bin/env python3
"""Size and power validation of the boundary-escape test.

The committed positive validation of the test is concentrated in one known-
defective dataset (SROIE). This script adds a controlled validation on clean
data, entirely post-hoc over the committed probability caches (no training):

1. **Size under a synthetic null.** For every dataset, hold the observed
   (L_s, k_s) structure of the informative spans fixed and redraw the escaped
   positions uniformly within each span -- the exchangeable-position null the
   test assumes -- then run the full pipeline (Monte-Carlo null, then Holm /
   BH / BY at gamma = 0.10) and record the family-wise rejection frequency
   over replicates. This measures the realized false-alarm rate of each
   deployed multiplicity procedure.

2. **Power under boundary-localized corruption.** For a target field type t on
   a clean dataset, corrupt a fraction eps of its calibration spans by
   extending the span end one token onto a following O token (relabel it I-t)
   -- the single-token boundary-overreach mechanism the SROIE audit surfaced:
   the annotation claims one token at a span edge that does not belong to the
   field, so exactly that edge token escapes. (A start-side shift that moves
   the B- tag corrupts two tokens -- the new edge and the demoted old B-, an
   interior position -- and therefore dilutes its own edge signature; we
   report it as a harder secondary variant.) Scores, the pooled CRC threshold
   tau, spans, and the test are all recomputed from the corrupted labels,
   exactly as a user of the pipeline would (30 corruption seeds per cell, so a
   power estimate has a binomial standard error of at most 0.09). Recorded per
   (field, eps, seed):
   rejection of the corrupted field under Holm / BH / BY (power), rejections
   among untouched fields (false positives) under each, every field p-value,
   enrichment, informative-span counts, and the corrupted field's rank under a
   raw per-field span-coverage-deficit screening (the baseline the paper
   argues lacks specificity).

Datasets: FUNSD and WildReceipt (clean negatives with an O background class,
so the corruption is well defined). CORD has no O class and is excluded from
the corruption arm; its null-size arm still runs.

Writes results/<tag>/boundary_escape_power.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_crc import (DEFAULT_CACHE, DEFAULT_RESULTS, bh_reject, holm_reject,
                      by_reject, build_spans,
                      boundary_escape_unit, calibration_units, crc_lambda,
                      load_cache, _by_doc)

GAMMA = 0.10
N_BOOT = 5000


def fast_crc_lambda(scores_by_doc, alpha):
    """Vectorized equivalent of span_crc.crc_lambda.

    Mean risk over docs is a right-continuous step function of lambda that
    drops by 1/(n*m_j) when lambda crosses a score of doc j; a single sorted
    sweep finds the smallest lambda with (n*R(lambda)+1)/(n+1) <= alpha.
    """
    n = len(scores_by_doc)
    if n == 0:
        return 1.0, 0.0
    target = (alpha * (n + 1) - 1.0) / n
    if target < 0:
        return 1.0, 0.0
    scores = np.concatenate(scores_by_doc)
    w = np.concatenate([np.full(len(d), 1.0 / (n * len(d)))
                        for d in scores_by_doc])
    order = np.argsort(scores, kind="stable")
    risk_after = 1.0 - np.cumsum(w[order])
    idx = np.searchsorted(-risk_after, -target, side="left")
    if idx >= len(scores):
        return 1.0, 0.0
    lam = float(scores[order][idx])
    # step function: equal scores must all be consumed
    while idx + 1 < len(scores) and scores[order][idx + 1] == lam:
        idx += 1
    return lam, float(risk_after[idx])
NULL_REPS = 200
DOSES = [0.02, 0.05, 0.10, 0.20, 0.50]
CORRUPT_SEEDS = list(range(30))
TARGETS = {
    "funsd": ["QUESTION", "ANSWER", "HEADER"],
    "wildreceipt": ["Store_addr_value", "Prod_item_value", "Date_value",
                    "Total_value"],
}


def first_unit(ds):
    z, meta, id2label = load_cache(ds, DEFAULT_CACHE)
    name, cal_key, _ = calibration_units(meta)[0]
    return (z[f"{cal_key}_probs"], np.array(z[f"{cal_key}_labels"]),
            z[f"{cal_key}_docstart"], id2label, name)


# ---------------------------------------------------------------- size arm

def null_size(ds, probs, labels, docstart, id2label):
    spans, lac = build_spans(probs, labels, docstart, id2label)
    tau, _ = fast_crc_lambda(_by_doc(spans), GAMMA)
    # collect the informative-span structure per field once
    struct = {}
    for s in spans:
        seg = lac[s["lo"]:s["hi"]]
        k, L = int((seg > tau).sum()), int(s["len"])
        if 0 < k < L:
            struct.setdefault(s["field"], []).append((L, k))
    struct = {f: v for f, v in struct.items() if v}
    rng = np.random.default_rng(12345)
    procs = {"holm": holm_reject, "bh": bh_reject, "by": by_reject}
    n_reject = {k: 0 for k in procs}
    per_field_rej = {k: {f: 0 for f in struct} for k in procs}
    n_uncorrected = 0
    for _ in range(NULL_REPS):
        pvals = {}
        for f, recs in struct.items():
            ne = 2
            obs = sum(int(rng.hypergeometric(min(ne, L), L - min(ne, L),
                                             min(k, L))) for L, k in recs)
            null = np.zeros(N_BOOT, dtype=np.int64)
            for L, k in recs:
                null += rng.hypergeometric(min(ne, L), L - min(ne, L),
                                           min(k, L), size=N_BOOT)
            pvals[f] = float((1 + int((null >= obs).sum())) / (N_BOOT + 1))
        if any(p <= GAMMA for p in pvals.values()):
            n_uncorrected += 1
        for name, fn in procs.items():
            rej = fn(pvals, GAMMA)
            if any(rej.values()):
                n_reject[name] += 1
            for f, r in rej.items():
                per_field_rej[name][f] += int(r)
    return {
        "n_replicates": NULL_REPS,
        "tau": float(tau),
        "n_informative_by_field": {f: len(v) for f, v in struct.items()},
        "familywise_rejection_rate": n_reject["holm"] / NULL_REPS,
        "familywise_rejection_rate_by_procedure":
            {k: v / NULL_REPS for k, v in n_reject.items()},
        "familywise_rejection_rate_uncorrected": n_uncorrected / NULL_REPS,
        "per_field_rejection_rate":
            {k: {f: c / NULL_REPS for f, c in d.items()}
             for k, d in per_field_rej.items()},
    }


# --------------------------------------------------------------- power arm

def corrupt_boundary(labels, docstart, id2label, label2id, field, eps, seed,
                     variant="end_extend"):
    """Boundary-localized corruption of a fraction eps of type-`field` spans.

    end_extend  : relabel the O token following the span as I-field (one
                  corrupted token, at the new end edge).
    start_shift : move the span start one token left onto a preceding O token
                  (B- moves; two corrupted tokens, edge + interior).
    Returns corrupted labels, #corrupted spans, #candidate spans."""
    lab = labels.copy()
    o_id = label2id["O"]
    b_id, i_id = label2id[f"B-{field}"], label2id[f"I-{field}"]
    dummy = np.ones((len(lab), max(id2label) + 1))
    spans, _ = build_spans(dummy, lab, docstart, id2label)  # only spans used
    doc_lo = {d: int(docstart[d]) for d in range(len(docstart) - 1)}
    doc_hi = {d: int(docstart[d + 1]) for d in range(len(docstart) - 1)}
    if variant == "end_extend":
        cand = [s for s in spans
                if s["field"] == field and s["hi"] < doc_hi[s["doc"]]
                and lab[s["hi"]] == o_id]
    else:
        cand = [s for s in spans
                if s["field"] == field and s["lo"] > doc_lo[s["doc"]]
                and lab[s["lo"] - 1] == o_id]
    rng = np.random.default_rng(seed)
    n_pick = int(round(eps * len(cand)))
    picked = rng.choice(len(cand), size=n_pick, replace=False) if n_pick else []
    for j in picked:
        s = cand[j]
        if variant == "end_extend":
            lab[s["hi"]] = i_id
        else:
            lab[s["lo"] - 1] = b_id
            lab[s["lo"]] = i_id
    return lab, n_pick, len(cand)


def power_cell(probs, labels, docstart, id2label, label2id, field, eps, seed,
               variant="end_extend"):
    lab_c, n_corr, n_cand = corrupt_boundary(labels, docstart, id2label,
                                             label2id, field, eps, seed,
                                             variant)
    spans, lac = build_spans(probs, lab_c, docstart, id2label)
    tau, _ = fast_crc_lambda(_by_doc(spans), GAMMA)
    be = boundary_escape_unit(spans, lac, probs, id2label, tau=tau,
                              n_boot=N_BOOT, seed=seed, gamma=GAMMA)
    tgt = be["fields"].get(field, {})
    allp = {f: d["p_value"] for f, d in be["fields"].items()
            if d["p_value"] is not None}
    dec = {"holm": holm_reject(allp, GAMMA), "bh": bh_reject(allp, GAMMA),
           "by": by_reject(allp, GAMMA)}
    fp = {k: sorted([f for f, r in v.items() if r and f != field])
          for k, v in dec.items()}
    # raw screening baseline: rank of the corrupted field by span-coverage
    # deficit under the pooled threshold (1 = worst-covered field)
    cov = {}
    for s in spans:
        cov.setdefault(s["field"], []).append(s["score"] <= tau)
    deficit = {f: 1.0 - float(np.mean(v)) for f, v in cov.items()}
    order = sorted(deficit, key=deficit.get, reverse=True)
    return {
        "eps": eps, "seed": seed,
        "n_corrupted": n_corr, "n_candidates": n_cand,
        "tau": float(tau),
        "detected": bool(dec["holm"].get(field, False)),
        "detected_by_procedure": {k: bool(v.get(field, False))
                                  for k, v in dec.items()},
        "all_pvalues": allp,
        "p_value": tgt.get("p_value"),
        "enrichment": tgt.get("edge_enrichment"),
        "n_informative": tgt.get("n_informative_spans", 0),
        "false_positive_fields": fp["holm"],
        "false_positive_fields_by_procedure": fp,
        "n_fields_tested": be["n_fields_tested"],
        "screening_rank_of_target": order.index(field) + 1 if field in order else None,
        "screening_n_fields_worse_or_equal": int(sum(
            deficit[f] >= deficit.get(field, 0.0) for f in deficit)),
    }


def main():
    out = {"gamma": GAMMA, "n_boot": N_BOOT, "doses": DOSES,
           "corrupt_seeds": CORRUPT_SEEDS, "null_replicates": NULL_REPS,
           "size": {}, "power": {}}
    for ds in ["funsd", "wildreceipt", "cord"]:
        probs, labels, docstart, id2label, unit = first_unit(ds)
        out["size"][ds] = null_size(ds, probs, labels, docstart, id2label)
        r = out["size"][ds]
        print(f"[{ds}] size arm (unit {unit}): familywise rejection "
              f"{r['familywise_rejection_rate']:.3f} over {NULL_REPS} null replicates")
    for variant in ("end_extend", "start_shift"):
        out["power"][variant] = {}
        for ds, fields in TARGETS.items():
            probs, labels, docstart, id2label, unit = first_unit(ds)
            label2id = {v: k for k, v in id2label.items()}
            out["power"][variant][ds] = {}
            for field in fields:
                cells = []
                for eps in DOSES:
                    for seed in CORRUPT_SEEDS:
                        cells.append(power_cell(probs, labels, docstart,
                                                id2label, label2id, field,
                                                eps, seed, variant))
                out["power"][variant][ds][field] = cells
                for eps in DOSES:
                    sub = [c for c in cells if c["eps"] == eps]
                    det = np.mean([c["detected"] for c in sub])
                    det_bh = np.mean([c["detected_by_procedure"]["bh"] for c in sub])
                    det_by = np.mean([c["detected_by_procedure"]["by"] for c in sub])
                    fp = np.mean([len(c["false_positive_fields"]) for c in sub])
                    inf = np.mean([c["n_informative"] for c in sub])
                    nc = sub[0]["n_corrupted"]
                    print(f"[{variant}][{ds}] {field:<18} eps={eps:.2f} "
                          f"(n={nc:>3}): power holm={det:.2f} bh={det_bh:.2f} "
                          f"by={det_by:.2f}  FP/run={fp:.2f}  "
                          f"informative={inf:.0f}")
    dst = os.path.join(DEFAULT_RESULTS, "boundary_escape_power.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
