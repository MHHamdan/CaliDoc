#!/usr/bin/env python3
"""Span-level conformal risk control (SpanCRC) and the boundary-escape diagnostic.

Motivation
----------
Every conformal result in this project calibrates on *tokens* while splitting on
*documents*. The guarantee that produces is real and, for key information
extraction, close to useless: on SROIE the marginal token threshold delivers
~0.894 token coverage and 0.000 span coverage over 420 gold spans, because the
threshold is sized by a background class holding ~85% of tokens.

This module changes the controlled object rather than the model. Two pieces:

1. **SpanCRC.** For a gold BIO span s = (a, b, t), define the span nonconformity
   score as the max token LAC score inside it,

       S(s) = max_{a <= i < b} (1 - p_i[g_i]),

   so that `span s is fully covered at lambda  <=>  S(s) <= lambda`. Span
   coverage is therefore a one-dimensional conformal problem. Spans inside a
   page are not exchangeable (one bad OCR line corrupts several at once), so we
   take the *document* as the exchangeable unit and control a risk rather than a
   coverage, following conformal risk control (Angelopoulos, Bates, Fisch, Lei,
   Schuster, "Conformal Risk Control", ICLR 2024, arXiv:2208.02814):

       R_j(lambda) = (1/|S_j|) sum_{s in S_j} 1[S(s) > lambda]
       lambda_hat  = inf { lambda : (n Rhat(lambda) + B) / (n + 1) <= alpha },  B = 1

   giving E[R_{n+1}(lambda_hat)] <= alpha -- the expected fraction of gold fields
   missed on a fresh document. Repeating per field type with a null-cluster
   fallback for low-support fields gives the field-conditional variant. Its
   prediction sets use the *candidate-label* rule: the set of token i is
   {y : 1 - p(y|x_i) <= lambda_{type(y)}}, where type(y) is the field of BIO
   label y and non-field labels use the pooled lambda. Under this rule a gold
   type-t span is covered iff S(s) <= lambda_t, so per-type CRC calibration
   matches the deployed rule exactly and the per-type guarantee (over
   calibration documents containing that type) follows directly; no threshold
   selection depends on test-time predictions. Reported set sizes are computed
   under this same rule.

2. **Boundary-escape test.** At a reference threshold tau (the pooled SpanCRC
   lambda, so no free knob), the escape set of a span is
   E(s) = { i in [a,b) : 1 - p_i[g_i] > tau }. A span is *informative* iff
   0 < |E(s)| < L_s.

   NULL (stated precisely). The test is a conditional randomization test.
   Conditional on the observed (L_s, k_s) of *every* informative type-t span,
   H_0^t asserts that each span's escaped-position set is uniform over the
   C(L_s, k_s) subsets of its positions, jointly independently across spans.
   The edge count of one span is then Hypergeometric(L_s, |B_s|, k_s) and W_t
   is their sum. Cross-span independence is an ASSUMPTION, not a consequence:
   spans inside one document can share a defect (this module's own CRC
   construction assumes documents, not spans, are exchangeable). We therefore
   also run a document-blocked (cluster) bootstrap of the enrichment ratio,
   resampling whole documents with replacement, which assumes independence
   only across documents and leaves within-document dependence arbitrary. Both
   are reported; a rejection is only claimed when the two agree.

   Multiplicity across fields is controlled by Holm-Bonferroni (FWER under
   arbitrary dependence -- the primary procedure, since it needs no dependence
   argument at all), with Benjamini-Hochberg (FDR under PRDS) and
   Benjamini-Yekutieli (FDR under arbitrary dependence) reported alongside.

   This is the specificity control the per-class quantile-spread statistic never
   had: it is support-adaptive by construction (few informative spans => wide
   null => no rejection) and difficulty-adaptive (a uniformly hard field escapes
   everywhere and yields enrichment 1.0).

Everything runs post-hoc on the cached temperature-scaled probabilities in
`results/cache/<ds>_local_20260803.npz`. No GPU, no retraining.

Self-check
----------
`--self_check` recomputes two quantities that already exist as committed
artifacts and refuses to write output if they disagree:
  * `conditional_cp_<ds>.json -> per_unit[0].alpha_0.1.o_class_analysis.span_coverage`
  * `ablations_<ds>.json      -> per_unit[0].entity_level_cp.alpha_0.1.entity_coverage`

Usage
-----
    python3 scripts/span_crc.py --dataset sroie
    python3 scripts/span_crc.py --all --results_dir results/local_20260803
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(REPO, "results", "cache")
DEFAULT_RESULTS = os.path.join(REPO, "results", "local_20260803")
CACHE_TAG = "local_20260803"

DATASETS = ["cord", "funsd", "sroie", "wildreceipt", "sroie_repaired",
            "wildreceipt_traincal"]


# --------------------------------------------------------------------------
# conformal primitives
# --------------------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample correction.

    Matches `evaluate_conformal_phase2.py` so the token baseline reproduces the
    committed numbers exactly.
    """
    n = len(scores)
    if n == 0:
        return 1.0
    q = math.ceil((n + 1) * (1 - alpha)) / n
    if q >= 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, q, method="higher"))


def crc_lambda(scores_by_doc: Sequence[np.ndarray], alpha: float) -> Tuple[float, float]:
    """Conformal risk control threshold with B = 1.

    Returns (lambda_hat, Rhat(lambda_hat)). The risk is non-increasing in lambda
    and R(1) = 0, so a solution always exists in the score grid.
    """
    if not len(scores_by_doc):
        return 1.0, 0.0
    grid = np.unique(np.concatenate(list(scores_by_doc) + [np.array([1.0])]))
    grid.sort()
    n = len(scores_by_doc)
    for lam in grid:
        risk = float(np.mean([(d > lam).mean() for d in scores_by_doc]))
        if (n * risk + 1.0) / (n + 1) <= alpha:
            return float(lam), risk
    return 1.0, 0.0


def bh_reject(pvals: Dict[str, float], gamma: float) -> Dict[str, bool]:
    """Benjamini-Hochberg at level gamma.

    Controls FDR under independence or positive regression dependence (PRDS).
    Field p-values here share documents, model and calibration split, so PRDS
    is an assumption, not a fact -- which is why `holm_reject` is the primary
    procedure and `by_reject` the dependence-agnostic FDR fallback.
    """
    if not pvals:
        return {}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    k_max = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= gamma * i / m:
            k_max = i
    keep = {k for k, _ in items[:k_max]}
    return {k: (k in keep) for k in pvals}


def holm_reject(pvals: Dict[str, float], gamma: float) -> Dict[str, bool]:
    """Holm-Bonferroni at level gamma.

    Controls the familywise error rate under *arbitrary* dependence among the
    p-values, so it needs no assumption about how field-level tests covary.
    """
    if not pvals:
        return {}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    keep = set()
    for i, (k, p) in enumerate(items, start=1):
        if p <= gamma / (m - i + 1):
            keep.add(k)
        else:
            break                      # step-down: stop at the first failure
    return {k: (k in keep) for k in pvals}


def by_reject(pvals: Dict[str, float], gamma: float) -> Dict[str, bool]:
    """Benjamini-Yekutieli at level gamma: FDR under arbitrary dependence."""
    if not pvals:
        return {}
    m = len(pvals)
    c_m = sum(1.0 / i for i in range(1, m + 1))
    return bh_reject(pvals, gamma / c_m)


def adjusted_pvalues(pvals: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Holm-, BH- and BY-adjusted p-values (monotone, capped at 1)."""
    if not pvals:
        return {}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    c_m = sum(1.0 / i for i in range(1, m + 1))
    holm, run = {}, 0.0
    for i, (k, p) in enumerate(items, start=1):        # step-down, cumulative max
        run = max(run, min(1.0, (m - i + 1) * p))
        holm[k] = run
    bh, run = {}, 1.0
    for i in range(m, 0, -1):                          # step-up, cumulative min
        k, p = items[i - 1]
        run = min(run, min(1.0, m * p / i))
        bh[k] = run
    by, run = {}, 1.0
    for i in range(m, 0, -1):
        k, p = items[i - 1]
        run = min(run, min(1.0, c_m * m * p / i))
        by[k] = run
    return {k: {"p": pvals[k], "holm": holm[k], "bh": bh[k], "by": by[k]}
            for k in pvals}


# --------------------------------------------------------------------------
# span extraction
# --------------------------------------------------------------------------

def spans_of(tags: Sequence[str]) -> List[Tuple[int, int]]:
    """Maximal BIO segments as [start, end) index pairs.

    Mirrors `_spans()` in `conditional_cp_baselines.py`: a bare I- tag opens a
    span (the datasets contain them), and B- always starts a new one.
    """
    out: List[Tuple[int, int]] = []
    start = None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if start is not None:
                out.append((start, i))
            start = i
        elif t.startswith("I-"):
            if start is None:
                start = i
        else:
            if start is not None:
                out.append((start, i))
                start = None
    if start is not None:
        out.append((start, len(tags)))
    return out


def build_spans(probs, labels, docstart, id2label):
    """Gold spans with their max-token LAC score, plus the token LAC vector."""
    lac = 1.0 - probs[np.arange(len(labels)), labels]
    out = []
    for d in range(len(docstart) - 1):
        s, e = int(docstart[d]), int(docstart[d + 1])
        tags = [id2label[int(t)] for t in labels[s:e]]
        for a, b in spans_of(tags):
            out.append({
                "doc": d,
                "field": tags[a][2:],
                "len": b - a,
                "score": float(lac[s + a:s + b].max()),
                "lo": s + a,
                "hi": s + b,
            })
    return out, lac


def load_cache(dataset: str, cache_dir: str, cache_tag: str = CACHE_TAG):
    z = np.load(os.path.join(cache_dir, f"{dataset}_{cache_tag}.npz"))
    meta = json.load(open(os.path.join(cache_dir, f"{dataset}_{cache_tag}.json")))
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    return z, meta, id2label


def calibration_units(meta) -> List[Tuple[str, str, str]]:
    """(name, cal_key, ev_key) for every calibration split in the cache."""
    units = meta["units"]
    if "cal" in units and "test" in units and not any(u.startswith("cal_s") for u in units):
        # CORD ships an independent calibration split; test is the eval side.
        return [("fixed", "cal", "test")]
    return [(u.replace("cal_", "seed_"), u, u.replace("cal_", "ev_"))
            for u in units if u.startswith("cal_") and u.replace("cal_", "ev_") in units]


def _by_doc(spans) -> List[np.ndarray]:
    d: Dict[int, List[float]] = {}
    for s in spans:
        d.setdefault(s["doc"], []).append(s["score"])
    return [np.asarray(v) for v in d.values()]


# --------------------------------------------------------------------------
# SpanCRC
# --------------------------------------------------------------------------

def span_crc_unit(cal_spans, ev_spans, ev_probs, ev_labels, cal_lac, ev_lac,
                  alpha: float, min_cal_spans: int, id2label=None) -> dict:
    """All thresholds and their evaluation for one calibration split."""
    qtok = conformal_quantile(cal_lac, alpha)
    token_cov = float((ev_lac <= qtok).mean())
    size_tok = float((ev_probs >= 1 - qtok).sum(1).mean())
    span_cov_tok = float(np.mean([s["score"] <= qtok for s in ev_spans]))

    lam0, rhat0 = crc_lambda(_by_doc(cal_spans), alpha)
    span_cov_pool = float(np.mean([s["score"] <= lam0 for s in ev_spans]))
    size_pool = float((ev_probs >= 1 - lam0).sum(1).mean())

    # field-conditional, with a null-cluster fallback to the pooled threshold.
    cal_fields: Dict[str, list] = {}
    for s in cal_spans:
        cal_fields.setdefault(s["field"], []).append(s)
    lam_f: Dict[str, float] = {}
    n_fallback = 0
    for f, group in cal_fields.items():
        if len(group) < min_cal_spans:
            lam_f[f] = lam0
            n_fallback += 1
            continue
        lam_f[f], _ = crc_lambda(_by_doc(group), alpha)

    span_cov_fc = float(np.mean([s["score"] <= lam_f.get(s["field"], lam0) for s in ev_spans]))
    # deployable candidate-label rule: candidate y is admitted at threshold
    # lambda_{type(y)}; non-field labels (O) use the pooled lambda. This is the
    # rule the guarantee is stated for, and the set size the paper reports.
    thr = np.full(ev_probs.shape[1], lam0, dtype=float)
    for j in range(ev_probs.shape[1]):
        lab = id2label[j]
        if lab.startswith(("B-", "I-")):
            thr[j] = lam_f.get(lab[2:], lam0)
    size_fc = float((ev_probs >= 1.0 - thr[None, :]).sum(1).mean())

    ev_fields = sorted({s["field"] for s in ev_spans})
    per_field = {}
    for f in ev_fields:
        fe = [s for s in ev_spans if s["field"] == f]
        per_field[f] = {
            "n_cal_spans": len(cal_fields.get(f, [])),
            "n_ev_spans": len(fe),
            "lambda": lam_f.get(f, lam0),
            "fallback": len(cal_fields.get(f, [])) < min_cal_spans,
            "span_cov_token_threshold": float(np.mean([s["score"] <= qtok for s in fe])),
            "span_cov_pooled_crc": float(np.mean([s["score"] <= lam0 for s in fe])),
            "span_cov_field_crc": float(np.mean([s["score"] <= lam_f.get(f, lam0) for s in fe])),
        }

    def covgap(key: str) -> float:
        vals = [v[key] for v in per_field.values()]
        return float(np.mean([abs(v - (1 - alpha)) for v in vals]) * 100) if vals else float("nan")

    # per-document realized risk -- the quantity CRC actually bounds. Reported
    # under the token-marginal threshold too, so the main table can show the
    # controlled quantity beside the empirical one.
    ev_docs = _by_doc(ev_spans)
    docs_fc = {}
    for s in ev_spans:
        docs_fc.setdefault(s["doc"], []).append(
            s["score"] > lam_f.get(s["field"], lam0))
    risk_fc = float(np.mean([np.mean(v) for v in docs_fc.values()]))
    return {
        "n_cal_spans": len(cal_spans),
        "n_ev_spans": len(ev_spans),
        "n_cal_docs_with_spans": len(_by_doc(cal_spans)),
        "n_fields_fallback": n_fallback,
        "token_marginal": {
            "qhat": qtok,
            "token_coverage": token_cov,
            "span_coverage": span_cov_tok,
            "avg_set_size": size_tok,
            "ev_doc_risk": float(np.mean([(d > qtok).mean() for d in ev_docs])),
            "span_covgap_pp": covgap("span_cov_token_threshold"),
        },
        "pooled_crc": {
            "lambda": lam0,
            "cal_risk": rhat0,
            "span_coverage": span_cov_pool,
            "avg_set_size": size_pool,
            "ev_doc_risk": float(np.mean([(d > lam0).mean() for d in ev_docs])),
            "span_covgap_pp": covgap("span_cov_pooled_crc"),
        },
        "field_crc": {
            "min_cal_spans": min_cal_spans,
            "span_coverage": span_cov_fc,
            "avg_set_size": size_fc,
            "ev_doc_risk": risk_fc,
            "span_covgap_pp": covgap("span_cov_field_crc"),
        },
        "per_field": per_field,
    }


# --------------------------------------------------------------------------
# boundary-escape diagnostic
# --------------------------------------------------------------------------

def position_mask(kind: str, L: int) -> np.ndarray:
    """Pre-specified position mask B_s for a span of length L.

    The conditional randomization test stays exact for ANY mask that is fixed
    independently of where the escapes landed: masks may depend on the span's
    identity, length, field type, or tau -- all invariant under the
    within-span permutation group -- but never on the escaped positions
    themselves. Post-hoc masks chosen after inspecting escape locations
    destroy the validity of the p-values. If more than one mask is tested,
    every (field, mask) pair must enter the multiplicity correction.
    """
    m = np.zeros(L, dtype=bool)
    if kind == "edges":              # historical default: first and last token
        m[0] = True
        m[-1] = True
    elif kind == "first":
        m[0] = True
    elif kind == "last":
        m[-1] = True
    elif kind == "interior":         # empty for L <= 2 (structural blind spot)
        if L > 2:
            m[1:-1] = True
    else:
        raise ValueError(f"unknown position mask '{kind}'")
    return m


def boundary_escape_unit(cal_spans, cal_lac, cal_probs, id2label, tau: float,
                         n_boot: int, seed: int, gamma: float,
                         mask: str = "edges") -> dict:
    """Positional-enrichment test on the calibration split at threshold tau.

    `mask` names a pre-specified position subset B_s (see `position_mask`);
    the null count for one span is Hypergeometric(L_s, |B_s|, k_s) and the
    default `edges` reproduces the paper's boundary-escape test exactly.
    """
    rng = np.random.default_rng(seed)
    # independent stream: the cluster bootstrap must not perturb the draws
    # that produce the Monte-Carlo p-values, so those stay reproducible.
    rng_boot = np.random.default_rng(seed + 10_000)
    by_field: Dict[str, List[Tuple[int, int, int, int]]] = {}
    escaped_idx: Dict[str, List[int]] = {}
    n_spans: Dict[str, int] = {}

    doc_of: Dict[str, List[int]] = {}
    for s in cal_spans:
        f = s["field"]
        n_spans[f] = n_spans.get(f, 0) + 1
        seg = cal_lac[s["lo"]:s["hi"]]
        esc = seg > tau
        k = int(esc.sum())
        L = int(s["len"])
        if k == 0 or k == L:          # non-informative: carries no positional signal
            continue
        edge = position_mask(mask, L)
        if not edge.any():            # mask degenerate for this length (e.g.
            continue                  # interior mask on L <= 2): no information
        by_field.setdefault(f, []).append((L, k, int((esc & edge).sum()), int(edge.sum())))
        doc_of.setdefault(f, []).append(int(s["doc"]))
        escaped_idx.setdefault(f, []).extend((s["lo"] + np.flatnonzero(esc)).tolist())

    fields = {}
    pvals = {}
    for f, recs in by_field.items():
        obs = sum(r[2] for r in recs)
        null = np.zeros(n_boot, dtype=np.int64)
        exp = 0.0
        for (L, k, _, ne) in recs:
            null += rng.hypergeometric(ne, L - ne, min(k, L), size=n_boot)
            exp += ne * min(k, L) / L
        p = float((1 + int((null >= obs).sum())) / (n_boot + 1))
        pvals[f] = p
        idx = escaped_idx.get(f, [])
        modal = None
        if idx:
            preds = cal_probs[np.asarray(idx)].argmax(1)
            vals, counts = np.unique(preds, return_counts=True)
            modal = {
                "label": id2label[int(vals[counts.argmax()])],
                "count": int(counts.max()),
                "n_escaped_tokens": int(len(idx)),
            }
        # document-blocked (cluster) bootstrap of the enrichment ratio: the
        # only independence assumed is across documents -- the same
        # exchangeable unit the CRC construction already uses. Arbitrary
        # within-document dependence between spans is preserved by resampling
        # whole documents.
        docs = np.asarray(doc_of.get(f, []))
        uniq = np.unique(docs)
        boot = None
        if exp > 0 and len(uniq) >= 2:
            obs_s = np.array([r[2] for r in recs], dtype=float)
            exp_s = np.array([r[3] * min(r[1], r[0]) / r[0] for r in recs],
                             dtype=float)
            idx_by_doc = [np.flatnonzero(docs == d) for d in uniq]
            ratios = np.empty(n_boot)
            for b in range(n_boot):
                pick = rng_boot.integers(0, len(uniq), len(uniq))
                sel = np.concatenate([idx_by_doc[j] for j in pick])
                e = exp_s[sel].sum()
                ratios[b] = obs_s[sel].sum() / e if e > 0 else np.nan
            ratios = ratios[np.isfinite(ratios)]
            if len(ratios):
                boot = {
                    "n_documents": int(len(uniq)),
                    "ci90_lo": float(np.quantile(ratios, 0.05)),
                    "ci90_hi": float(np.quantile(ratios, 0.95)),
                    "excludes_one_above": bool(np.quantile(ratios, 0.05) > 1.0),
                }
        fields[f] = {
            "n_spans": n_spans.get(f, 0),
            "n_informative_spans": len(recs),
            "n_documents_informative": int(len(uniq)),
            "edge_observed": obs,
            "edge_expected": float(exp),
            "edge_enrichment": float(obs / exp) if exp > 0 else None,
            "p_value": p,
            "cluster_bootstrap": boot,
            "modal_escape_prediction": modal,
        }

    adj = adjusted_pvalues(pvals)
    dec = {"holm": holm_reject(pvals, gamma),
           "bh": bh_reject(pvals, gamma),
           "by": by_reject(pvals, gamma)}
    for f in pvals:
        fields[f]["adjusted"] = adj[f]
        fields[f]["holm_rejected"] = bool(dec["holm"][f])
        fields[f]["bh_rejected"] = bool(dec["bh"][f])
        fields[f]["by_rejected"] = bool(dec["by"][f])
    for f in n_spans:
        if f not in fields:
            fields[f] = {"n_spans": n_spans[f], "n_informative_spans": 0,
                         "n_documents_informative": 0,
                         "edge_observed": 0, "edge_expected": 0.0,
                         "edge_enrichment": None, "p_value": None,
                         "cluster_bootstrap": None,
                         "modal_escape_prediction": None, "adjusted": None,
                         "holm_rejected": False, "bh_rejected": False,
                         "by_rejected": False}
    return {
        "tau": tau,
        "n_boot": n_boot,
        "gamma": gamma,
        "position_mask": mask,
        "primary_procedure": "holm",
        "null": ("product conditional randomization: conditional on every "
                 "observed (L_s, k_s), each informative span's escaped-position "
                 "set is uniform over the C(L_s, k_s) subsets, independently "
                 "across spans. Cross-span independence is an assumption; the "
                 "document-blocked cluster bootstrap in cluster_bootstrap "
                 "relaxes it to independence across documents only."),
        "n_fields_tested": len(pvals),
        "n_fields_rejected": int(sum(dec["holm"].values())),
        "rejected_fields": sorted([f for f, r in dec["holm"].items() if r]),
        "rejected_fields_bh": sorted([f for f, r in dec["bh"].items() if r]),
        "rejected_fields_by": sorted([f for f, r in dec["by"].items() if r]),
        "fields": fields,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_dataset(dataset: str, cache_dir: str, alpha: float, min_cal_spans: int,
                n_boot: int, seed: int, gamma: float,
                cache_tag: str = CACHE_TAG) -> dict:
    z, meta, id2label = load_cache(dataset, cache_dir, cache_tag)
    units = calibration_units(meta)
    if not units:
        raise RuntimeError(f"{dataset}: no calibration units in cache")

    per_unit = []
    for name, cal_key, ev_key in units:
        cal_p, cal_y, cal_ds = z[f"{cal_key}_probs"], z[f"{cal_key}_labels"], z[f"{cal_key}_docstart"]
        ev_p, ev_y, ev_ds = z[f"{ev_key}_probs"], z[f"{ev_key}_labels"], z[f"{ev_key}_docstart"]
        cal_spans, cal_lac = build_spans(cal_p, cal_y, cal_ds, id2label)
        ev_spans, ev_lac = build_spans(ev_p, ev_y, ev_ds, id2label)
        if not cal_spans or not ev_spans:
            continue
        res = span_crc_unit(cal_spans, ev_spans, ev_p, ev_y, cal_lac, ev_lac,
                            alpha, min_cal_spans, id2label)
        res["unit"] = name
        res["boundary_escape"] = boundary_escape_unit(
            cal_spans, cal_lac, cal_p, id2label,
            tau=res["pooled_crc"]["lambda"], n_boot=n_boot, seed=seed, gamma=gamma)
        per_unit.append(res)

    def agg(path: Sequence[str]) -> dict:
        vals = []
        for u in per_unit:
            node = u
            for k in path:
                node = node[k]
            vals.append(float(node))
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "values": vals, "n_units": len(vals)}

    return {
        "dataset": dataset,
        "protocol": {
            "alpha": alpha,
            "min_cal_spans": min_cal_spans,
            "n_boot": n_boot,
            "seed": seed,
            "bh_gamma": gamma,
            # stored relative to the repository root: an absolute path would
            # bake whoever ran it into the artifact.
            "cache": os.path.relpath(
                os.path.join(cache_dir, f"{dataset}_{cache_tag}.npz"), REPO),
            "temperature": meta["temperature"],
            "temperature_fit_split": meta["temperature_fit_split"],
            "exchangeable_unit": "document",
            "note": (
                "Span score is the max token LAC inside the gold span, so span "
                "coverage is exactly {S(s) <= lambda}. CRC (B=1) controls the "
                "expected per-document fraction of missed spans. Fields with "
                "fewer than min_cal_spans calibration spans fall back to the "
                "pooled threshold. Field-conditional sets use the "
                "candidate-label rule (threshold indexed by the candidate's "
                "field type), so calibration matches deployment and set sizes "
                "are reported under the deployable rule."
            ),
        },
        "aggregate": {
            "span_cov_token_marginal": agg(["token_marginal", "span_coverage"]),
            "span_cov_pooled_crc": agg(["pooled_crc", "span_coverage"]),
            "span_cov_field_crc": agg(["field_crc", "span_coverage"]),
            "span_covgap_pp_token_marginal": agg(["token_marginal", "span_covgap_pp"]),
            "span_covgap_pp_field_crc": agg(["field_crc", "span_covgap_pp"]),
            "avg_set_size_token_marginal": agg(["token_marginal", "avg_set_size"]),
            "avg_set_size_field_crc": agg(["field_crc", "avg_set_size"]),
            "n_fields_rejected": agg(["boundary_escape", "n_fields_rejected"]),
            "ev_doc_risk_token_marginal": agg(["token_marginal", "ev_doc_risk"]),
            "ev_doc_risk_field_crc": agg(["field_crc", "ev_doc_risk"]),
        },
        "per_unit": per_unit,
    }


def self_check(dataset: str, out: dict, results_dir: str) -> List[str]:
    """Compare against committed artifacts. Returns a list of failures."""
    fails = []
    u0 = out["per_unit"][0]

    p = os.path.join(results_dir, f"conditional_cp_{dataset}.json")
    if os.path.exists(p):
        try:
            ref = json.load(open(p))["per_unit"][0]["alpha_0.1"]["o_class_analysis"]["span_coverage"]
            got = u0["token_marginal"]["span_coverage"]
            if abs(float(ref) - got) > 1e-4:
                fails.append(f"span_coverage(token-marginal) {got:.6f} != "
                             f"conditional_cp_{dataset}.json o_class_analysis {float(ref):.6f}")
        except (KeyError, IndexError):
            pass

    p = os.path.join(results_dir, f"ablations_{dataset}.json")
    if os.path.exists(p):
        try:
            ref = json.load(open(p))["per_unit"][0]["entity_level_cp"]["alpha_0.1"]["entity_coverage"]
            got = u0["token_marginal"]["span_coverage"]
            if abs(float(ref) - got) > 1e-4:
                fails.append(f"span_coverage(token-marginal) {got:.6f} != "
                             f"ablations_{dataset}.json entity_coverage {float(ref):.6f}")
        except (KeyError, IndexError):
            pass
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--all", action="store_true", help="run every dataset")
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE)
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--min_cal_spans", type=int, default=25,
                    help="null-cluster fallback threshold (fields below this use "
                         "the pooled lambda)")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bh_gamma", type=float, default=0.10)
    ap.add_argument("--cache_tag", default=CACHE_TAG,
                    help="cache filename tag (e.g. base_local_20260815)")
    ap.add_argument("--self_check", action="store_true", default=True)
    ap.add_argument("--no_self_check", dest="self_check", action="store_false")
    ap.add_argument("--dry_run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    targets = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not targets:
        ap.error("pass --dataset <name> or --all")

    rc = 0
    for ds in targets:
        out = run_dataset(ds, args.cache_dir, args.alpha, args.min_cal_spans,
                          args.n_boot, args.seed, args.bh_gamma,
                          cache_tag=args.cache_tag)
        fails = self_check(ds, out, args.results_dir) if args.self_check else []
        out["self_check"] = {"passed": not fails, "failures": fails}

        a = out["aggregate"]
        print(f"\n=== {ds.upper()} ({len(out['per_unit'])} calibration unit(s)) ===")
        print(f"  span coverage  token-marginal : {a['span_cov_token_marginal']['mean']:.4f} "
              f"+- {a['span_cov_token_marginal']['std']:.4f}")
        print(f"  span coverage  pooled  CRC    : {a['span_cov_pooled_crc']['mean']:.4f} "
              f"+- {a['span_cov_pooled_crc']['std']:.4f}")
        print(f"  span coverage  field   CRC    : {a['span_cov_field_crc']['mean']:.4f} "
              f"+- {a['span_cov_field_crc']['std']:.4f}")
        print(f"  span CovGap pp  token -> field: {a['span_covgap_pp_token_marginal']['mean']:.2f}"
              f" -> {a['span_covgap_pp_field_crc']['mean']:.2f}")
        print(f"  avg set size    token -> field: {a['avg_set_size_token_marginal']['mean']:.3f}"
              f" -> {a['avg_set_size_field_crc']['mean']:.3f}")
        be = out["per_unit"][0]["boundary_escape"]
        print(f"  boundary-escape (unit 0, tau={be['tau']:.4f}): "
              f"{be['n_fields_rejected']}/{be['n_fields_tested']} fields rejected "
              f"(Holm, FWER {args.bh_gamma:.2f}) -> {be['rejected_fields']}")
        for f in be["rejected_fields"]:
            s = be["fields"][f]
            modal = s["modal_escape_prediction"]
            print(f"      {f:<12} enrich={s['edge_enrichment']:.2f} p={s['p_value']:.4f} "
                  f"modal={modal['label'] if modal else '-'}")
        if fails:
            rc = 1
            print("  SELF-CHECK FAILED:")
            for f in fails:
                print(f"    ! {f}")
        else:
            print("  self-check: PASS (reproduces committed artifacts)")

        if not args.dry_run:
            if fails:
                print(f"  refusing to write {ds}: self-check failed")
                continue
            os.makedirs(args.results_dir, exist_ok=True)
            dst = os.path.join(args.results_dir, f"span_crc_{ds}.json")
            with open(dst, "w") as fh:
                json.dump(out, fh, indent=2)
            print(f"  wrote {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
