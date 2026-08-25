#!/usr/bin/env python3
"""
Conditional-coverage baselines and controls (paper experiments E3 + E5).

Answers the two most credible reviewer attacks on the "per-class conformal
coverage audits labels" claim:

  E3  "Class-conditional CP is known to be hard when per-class calibration
       counts are small -- is your collapse just small-n?"
      -> per-class feasibility bound, matched-n subsampling control on a CLEAN
         dataset, clustered conformal prediction, and threshold interpolation
         between marginal and class-conditional calibration. If a clean class
         with the SAME tiny calibration count covers fine while the suspect
         class does not, the collapse is not a sample-size artifact.

  E5  "Your marginal coverage is carried entirely by the O class."
      -> coverage over all tokens / non-O tokens / BIO spans, plus the
         achievable-marginal-coverage curve as a function of the O base rate.
         Owns the finding instead of letting a reviewer weaponise it.

Everything here is post-hoc on cached probabilities (scripts/cache_probs.py),
so it is CPU-only and runs in seconds.

Metric note: CovGap is the average absolute deviation of class-conditional
coverage from the nominal level, in percentage points, following Ding et al.,
"Class-Conditional Conformal Prediction with Many Classes", NeurIPS 2023
(arXiv:2306.09335). Classes with no evaluation tokens are excluded.

Usage:
  python scripts/conditional_cp_baselines.py \
      --cache results/cache/cord_local_20260803.npz \
      --output results/local_20260803/conditional_cp_cord.json
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_conformal_phase2 import conformal_quantile, lac_scores

logger = logging.getLogger(__name__)

ALPHAS = (0.10, 0.05, 0.01)
CLUSTER_K_GRID = (2, 3, 4, 5, 8)
INTERP_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MATCHED_N_TRIALS = 20


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def covgap(per_class, alpha):
    """Average |coverage_y - (1-alpha)| over classes present in eval, in pp."""
    devs = [abs(v['coverage'] - (1 - alpha)) for v in per_class.values()
            if v.get('n_eval', 0) > 0]
    return float(np.mean(devs) * 100) if devs else None


def conditional_summary(per_class, alpha):
    covs = [(k, v['coverage']) for k, v in per_class.items() if v.get('n_eval', 0) > 0]
    if not covs:
        return {}
    vals = np.array([c for _, c in covs])
    worst_k, worst_v = min(covs, key=lambda t: t[1])
    return {
        'covgap_pp': covgap(per_class, alpha),
        'macro_coverage': float(vals.mean()),
        'worst_class': worst_k,
        'worst_class_coverage': float(worst_v),
        'n_classes_evaluated': int(len(vals)),
        'n_classes_below_nominal': int((vals < (1 - alpha)).sum()),
        'n_classes_below_half_nominal': int((vals < 0.5 * (1 - alpha)).sum()),
    }


def feasibility_bound(alpha):
    """Smallest per-class calibration count that admits a finite class-
    conditional quantile at level alpha: n >= ceil((1-alpha)/alpha) - 1.
    Below this, the class-conditional set is vacuous (all classes)."""
    return int(math.ceil((1 - alpha) / alpha) - 1)


# --------------------------------------------------------------------------
# Calibration strategies -> per-token thresholds
# --------------------------------------------------------------------------

def marginal_qhat(cal_probs, cal_labels, alpha):
    return conformal_quantile(lac_scores(cal_probs, cal_labels), alpha)


def classwise_qhat(cal_probs, cal_labels, alpha, classes):
    """Mondrian: one quantile per class. Classes absent from calibration get a
    vacuous threshold (1.0 -> the full label set)."""
    out = {}
    for c in classes:
        m = cal_labels == c
        out[c] = (conformal_quantile(lac_scores(cal_probs[m], cal_labels[m]), alpha)
                  if m.any() else 1.0)
    return out


def clustered_qhat(cal_probs, cal_labels, alpha, classes, n_clusters, seed=42):
    """Clustered conformal prediction (Ding et al., NeurIPS 2023).

    Rare classes (fewer calibration points than the feasibility bound) are put
    in a 'null cluster' that falls back to the marginal quantile. The remaining
    classes are embedded by a vector of nonconformity-score quantiles and
    k-means clustered; each cluster gets one pooled quantile.
    """
    from sklearn.cluster import KMeans

    n_min = feasibility_bound(alpha)
    scores_by_class, big = {}, []
    for c in classes:
        m = cal_labels == c
        s = lac_scores(cal_probs[m], cal_labels[m]) if m.any() else np.array([])
        scores_by_class[c] = s
        if len(s) > n_min:
            big.append(c)

    marg = marginal_qhat(cal_probs, cal_labels, alpha)
    qhat = {c: marg for c in classes}          # null-cluster default
    if len(big) < 2:
        return qhat, {'n_clustered': len(big), 'note': 'too few eligible classes'}

    k = int(min(n_clusters, len(big)))
    levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    emb = np.array([[np.quantile(scores_by_class[c], q) for q in levels] for c in big])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)

    for cl in range(k):
        members = [c for c, lab in zip(big, km.labels_) if lab == cl]
        pooled = np.concatenate([scores_by_class[c] for c in members])
        q = conformal_quantile(pooled, alpha)
        for c in members:
            qhat[c] = q
    return qhat, {
        'n_clusters': k,
        'n_clustered_classes': len(big),
        'n_null_cluster_classes': len(classes) - len(big),
        'feasibility_bound_n': n_min,
    }


def interpolated_qhat(cal_probs, cal_labels, alpha, classes, lam):
    """Threshold interpolation between marginal (lam=0) and class-conditional
    (lam=1) calibration -- our implementation of the marginal<->class-
    conditional interpolation family described for long-tailed conformal
    prediction (arXiv:2507.06867). Not a verbatim reproduction of that paper."""
    marg = marginal_qhat(cal_probs, cal_labels, alpha)
    cw = classwise_qhat(cal_probs, cal_labels, alpha, classes)
    return {c: (1 - lam) * marg + lam * cw[c] for c in classes}


def evaluate_qhat(ev_probs, ev_labels, qhat, id2label, classes):
    """Apply per-TRUE-class thresholds. Coverage for class y uses q_y; set size
    is reported under the same threshold."""
    per_class, covered_all, sizes_all = {}, [], []
    for c in classes:
        m = ev_labels == c
        if not m.any():
            per_class[id2label[c]] = {'n_eval': 0}
            continue
        q = qhat[c]
        in_set = ev_probs[m] >= (1.0 - q)
        cov = in_set[np.arange(m.sum()), ev_labels[m]]
        per_class[id2label[c]] = {
            'n_eval': int(m.sum()),
            'qhat': float(q),
            'coverage': float(cov.mean()),
            'avg_set_size': float(in_set.sum(axis=1).mean()),
        }
        covered_all.append(cov)
        sizes_all.append(in_set.sum(axis=1))
    covered = np.concatenate(covered_all)
    sizes = np.concatenate(sizes_all)
    return {
        'marginal_coverage': float(covered.mean()),
        'avg_set_size': float(sizes.mean()),
        'per_class': per_class,
    }


# --------------------------------------------------------------------------
# E3: matched-n control
# --------------------------------------------------------------------------

def matched_n_control(cal_probs, cal_labels, ev_probs, ev_labels, id2label,
                      classes, alpha, targets, trials=MATCHED_N_TRIALS, seed=42):
    """For each target calibration count n, subsample EVERY class that has at
    least n calibration tokens down to exactly n, recompute class-conditional
    coverage, and report the distribution. If coverage stays near nominal at
    the same n at which a suspect class collapsed, small-n is not the cause."""
    rng = np.random.RandomState(seed)
    out = {}

    # Fixed cohort: the classes eligible at the LARGEST target n. Subsampling
    # this same cohort across every n isolates the effect of calibration count
    # from the identity of the classes -- without it, growing n silently
    # restricts the average to the well-populated (and typically clean)
    # classes, which would manufacture the result we are testing for.
    n_max = max(targets)
    cohort = [c for c in classes
              if (cal_labels == c).sum() >= n_max and (ev_labels == c).any()]
    out['_fixed_cohort'] = {
        'classes': [id2label[c] for c in cohort],
        'n_classes': len(cohort),
        'defined_at_n': n_max,
        'note': ('Same classes at every n; compare against the collapsed '
                 'classes to separate small-n from label noise.'),
    }
    for n in targets:
        if cohort:
            covs = []
            for _ in range(trials):
                per_trial = []
                for c in cohort:
                    idx = np.where(cal_labels == c)[0]
                    pick = rng.choice(idx, size=n, replace=False)
                    q = conformal_quantile(
                        lac_scores(cal_probs[pick], cal_labels[pick]), alpha)
                    m = ev_labels == c
                    in_set = ev_probs[m] >= (1.0 - q)
                    per_trial.append(in_set[np.arange(m.sum()), ev_labels[m]].mean())
                covs.append(float(np.mean(per_trial)))
            out[f'fixed_cohort_n_{n}'] = {
                'mean_class_conditional_coverage': float(np.mean(covs)),
                'std': float(np.std(covs)),
                'min': float(np.min(covs)),
            }

        eligible = [c for c in classes
                    if (cal_labels == c).sum() >= n and (ev_labels == c).any()]
        if not eligible:
            out[f'n_{n}'] = {'n_eligible_classes': 0}
            continue
        covs = []
        for _ in range(trials):
            per_trial = []
            for c in eligible:
                idx = np.where(cal_labels == c)[0]
                pick = rng.choice(idx, size=n, replace=False)
                q = conformal_quantile(lac_scores(cal_probs[pick], cal_labels[pick]), alpha)
                m = ev_labels == c
                in_set = ev_probs[m] >= (1.0 - q)
                per_trial.append(in_set[np.arange(m.sum()), ev_labels[m]].mean())
            covs.append(float(np.mean(per_trial)))
        out[f'n_{n}'] = {
            'n_eligible_classes': len(eligible),
            'mean_class_conditional_coverage': float(np.mean(covs)),
            'std': float(np.std(covs)),
            'min': float(np.min(covs)),
            'trials': trials,
            'feasible': n >= feasibility_bound(alpha),
        }
    return out


# --------------------------------------------------------------------------
# E5: O-class ablation and span-level coverage
# --------------------------------------------------------------------------

def _spans(tags):
    spans, start = [], None
    for i, t in enumerate(tags):
        if t.startswith('B-'):
            if start is not None:
                spans.append((start, i))
            start = i
        elif t.startswith('I-'):
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i)); start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def o_class_analysis(cal_probs, cal_labels, ev_probs, ev_labels, docstart,
                     id2label, alpha):
    """Marginal coverage decomposed into the O and non-O parts, plus span-level
    coverage under a single marginal threshold."""
    o_ids = [i for i, name in id2label.items() if name == 'O']
    q = marginal_qhat(cal_probs, cal_labels, alpha)
    in_set = ev_probs >= (1.0 - q)
    covered = in_set[np.arange(len(ev_labels)), ev_labels]

    is_o = np.isin(ev_labels, o_ids) if o_ids else np.zeros(len(ev_labels), bool)
    p0 = float(is_o.mean())
    cov_o = float(covered[is_o].mean()) if is_o.any() else None
    cov_non_o = float(covered[~is_o].mean()) if (~is_o).any() else None

    # span level: a gold span is covered iff every token in it is covered
    n_span = n_cov = 0
    for d in range(len(docstart) - 1):
        s, e = docstart[d], docstart[d + 1]
        tags = [id2label[int(t)] for t in ev_labels[s:e]]
        for a, b in _spans(tags):
            n_span += 1
            if covered[s:e][a:b].all():
                n_cov += 1

    # Achievable marginal coverage if O were covered perfectly and every
    # non-O class covered at rate c: marginal = p0 + (1-p0) * c.
    curve = [{'non_o_coverage': float(c),
              'implied_marginal_at_observed_p0': float(p0 + (1 - p0) * c)}
             for c in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]

    return {
        'alpha': alpha,
        'qhat': q,
        'o_base_rate': p0,
        'marginal_coverage': float(covered.mean()),
        'o_coverage': cov_o,
        'non_o_coverage': cov_non_o,
        'span_coverage': (n_cov / n_span) if n_span else None,
        'n_spans': n_span,
        'marginal_minus_non_o_pp': (float((covered.mean() - cov_non_o) * 100)
                                    if cov_non_o is not None else None),
        'achievable_marginal_curve': curve,
        'note': ('With O base rate p0, marginal coverage >= 1-alpha is '
                 'attainable while every non-O class covers at c = '
                 '(1-alpha-p0)/(1-p0). Marginal coverage is therefore not a '
                 'reliability certificate for BIO tasks with dominant O.'),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_cache(npz_path):
    npz = np.load(npz_path)
    meta = json.loads(Path(str(npz_path).replace('.npz', '.json')).read_text())
    id2label = {int(k): v for k, v in meta['id2label'].items()}
    return npz, meta, id2label


def units_of(meta):
    """(name, cal_prefix, eval_prefix) triples matching the paper protocol."""
    if meta['dataset'] == 'cord':
        return [('fixed_split', 'cal', 'test')]
    return [(u.replace('cal_', 'seed_'), u, u.replace('cal_', 'ev_'))
            for u in meta['units'] if u.startswith('cal_')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--matched_n', default='5,10,20,50,100')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    npz, meta, id2label = load_cache(args.cache)
    targets = [int(x) for x in args.matched_n.split(',')]

    per_unit = []
    for name, cal_p, ev_p in units_of(meta):
        cal_probs, cal_labels = npz[f'{cal_p}_probs'], npz[f'{cal_p}_labels']
        ev_probs, ev_labels = npz[f'{ev_p}_probs'], npz[f'{ev_p}_labels']
        docstart = npz[f'{ev_p}_docstart']
        classes = sorted(set(np.unique(cal_labels)) | set(np.unique(ev_labels)))
        logger.info("unit %s: %d cal / %d eval tokens, %d classes",
                    name, len(cal_labels), len(ev_labels), len(classes))

        entry = {'unit': name, 'n_cal_tokens': int(len(cal_labels)),
                 'n_eval_tokens': int(len(ev_labels))}
        for alpha in ALPHAS:
            key = f'alpha_{alpha}'
            strategies = {}

            marg = {c: marginal_qhat(cal_probs, cal_labels, alpha) for c in classes}
            strategies['marginal'] = evaluate_qhat(ev_probs, ev_labels, marg,
                                                   id2label, classes)
            cw = classwise_qhat(cal_probs, cal_labels, alpha, classes)
            strategies['classwise_mondrian'] = evaluate_qhat(ev_probs, ev_labels, cw,
                                                             id2label, classes)
            for k in CLUSTER_K_GRID:
                q, info = clustered_qhat(cal_probs, cal_labels, alpha, classes, k)
                r = evaluate_qhat(ev_probs, ev_labels, q, id2label, classes)
                r['clustering'] = info
                strategies[f'clustered_k{k}'] = r
            for lam in INTERP_LAMBDAS:
                q = interpolated_qhat(cal_probs, cal_labels, alpha, classes, lam)
                strategies[f'interpolated_lambda{lam}'] = evaluate_qhat(
                    ev_probs, ev_labels, q, id2label, classes)

            for s in strategies.values():
                s['conditional_summary'] = conditional_summary(s['per_class'], alpha)

            entry[key] = {
                'feasibility_bound_n': feasibility_bound(alpha),
                'per_class_cal_counts': {
                    id2label[c]: int((cal_labels == c).sum()) for c in classes},
                'n_classes_below_feasibility': int(sum(
                    1 for c in classes
                    if (cal_labels == c).sum() < feasibility_bound(alpha))),
                'strategies': strategies,
                'matched_n_control': matched_n_control(
                    cal_probs, cal_labels, ev_probs, ev_labels, id2label,
                    classes, alpha, targets),
                'o_class_analysis': o_class_analysis(
                    cal_probs, cal_labels, ev_probs, ev_labels, docstart,
                    id2label, alpha),
            }
        per_unit.append(entry)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'dataset': meta['dataset'],
        'model_path': meta['model_path'],
        'temperature': meta['temperature'],
        'metric_definitions': {
            'covgap_pp': ('mean over evaluated classes of '
                          '|coverage_y - (1-alpha)|, in percentage points '
                          '(Ding et al., NeurIPS 2023, arXiv:2306.09335)'),
            'feasibility_bound_n': 'ceil((1-alpha)/alpha) - 1',
            'interpolated_lambda': ('our implementation of marginal<->class-'
                                    'conditional threshold interpolation for '
                                    'long-tailed CP (arXiv:2507.06867); not a '
                                    'verbatim reproduction'),
        },
        'per_unit': per_unit,
    }, indent=2))
    logger.info("Wrote %s", out)


if __name__ == '__main__':
    main()
