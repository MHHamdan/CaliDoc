#!/usr/bin/env python3
"""
Phase 2 conformal evaluation — the ONLY producer of results/phase2_results*.json,
which is the ONLY source for paper number updates (later phase).

Protocol (clean splits, enforced by calidoc.data.bundle guards):
- CORD : temperature on val_es -> score quantiles on cal -> evaluate test.
         Cross-conformal K=5 over the 50 cal docs as a secondary check
         (divergence > 2pp from the fixed split is a red flag).
- FUNSD: temperature on es_holdout -> for each persisted seed split
         {42,123,456}: quantiles on its 35-doc calibration subset of test,
         evaluate on its 15-doc evaluation subset; report mean±std.

Scores: LAC (1 - p_true) and APS (deterministic, no tie randomization —
recorded in the output metadata). Alphas: 0.10 / 0.05 / 0.01.
Per-class (Mondrian) analysis: CORD is non-O-only BY CONSTRUCTION (CORD gold
has no O class; the Phase 1 "O ≈ 5%" narrative was the label-schema artifact).

Usage:
  python scripts/evaluate_conformal_phase2.py \
      --model_path /scratch/hamdan/CaliDoc/outputs/<EXP>/best \
      --dataset cord --bundle_dir /scratch/hamdan/CaliDoc/data/data_bundle \
      --output /scratch/hamdan/CaliDoc/results/phase2_results_cord.json
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from calidoc.data import create_dataloader
from calidoc.data.bundle import FUNSD_CALIB_SEEDS, conformal_split
from calidoc.data.cord_dataset import CORDDataset
from calidoc.data.funsd_dataset import FUNSDDataset
from calidoc.data.wildreceipt_dataset import WildReceiptDataset
from calidoc.data.sroie_dataset import SROIEDataset
from calidoc.models import CaliDocModel

logger = logging.getLogger(__name__)

ALPHAS = (0.10, 0.05, 0.01)
CROSS_CONFORMAL_K = 5
CROSS_CONFORMAL_RED_FLAG_PP = 2.0


# --------------------------------------------------------------------------
# Logit collection
# --------------------------------------------------------------------------

def collect_doc_logits(model, dataset, device, batch_size):
    """Forward the whole dataset; return per-doc (logits, labels) for tokens
    with label != -100, in dataset order (dataset.doc_ids gives the ids)."""
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    per_doc = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(
                images=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                bbox=batch['bbox'],
                labels=None,
                return_loss=False,
            )
            logits = out['logits'].float().cpu()          # [B, L, C]
            labels = batch['labels'].cpu()                # [B, L]
            for i in range(logits.shape[0]):
                mask = labels[i] != -100
                per_doc.append((logits[i][mask], labels[i][mask]))
    assert len(per_doc) == len(dataset), (len(per_doc), len(dataset))
    return per_doc


def flatten(per_doc):
    logits = torch.cat([l for l, _ in per_doc])
    labels = torch.cat([y for _, y in per_doc])
    return logits, labels


def compute_test_metrics(per_doc, id2label, temperature, n_bins=15):
    """Point-prediction test-set metrics on the FULL test split:
    seqeval entity-level P/R/F1 (argmax predictions) + expected/max calibration
    error on temperature-scaled max-probability confidences. Reported alongside
    the conformal coverage so the paper's NER and ECE tables use the same
    corrected pipeline as the UQ tables."""
    from seqeval.metrics import precision_score, recall_score, f1_score

    true_seqs, pred_seqs = [], []
    all_conf, all_correct = [], []
    for logits, labels in per_doc:
        probs = F.softmax(logits / temperature, dim=-1).numpy()
        preds = probs.argmax(axis=1)
        y = labels.numpy()
        true_seqs.append([id2label[int(t)] for t in y])
        pred_seqs.append([id2label[int(p)] for p in preds])
        all_conf.append(probs.max(axis=1))
        all_correct.append((preds == y).astype(float))

    conf = np.concatenate(all_conf)
    correct = np.concatenate(all_correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += (m.mean()) * gap
        mce = max(mce, gap)
    return {
        'f1': float(f1_score(true_seqs, pred_seqs)),
        'precision': float(precision_score(true_seqs, pred_seqs)),
        'recall': float(recall_score(true_seqs, pred_seqs)),
        'token_accuracy': float(correct.mean()),
        'ece': float(ece * 100),
        'mce': float(mce * 100),
        'n_docs': len(per_doc),
        'n_tokens': int(len(conf)),
    }


# --------------------------------------------------------------------------
# Temperature scaling (fit ONLY on the ES split)
# --------------------------------------------------------------------------

def fit_temperature(logits, labels):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    logger.info("Fitted temperature: %.4f", t)
    return t


# --------------------------------------------------------------------------
# Conformal scores and sets
# --------------------------------------------------------------------------

def probs_of(logits, temperature):
    return F.softmax(logits / temperature, dim=-1).numpy()


def lac_scores(probs, labels):
    return 1.0 - probs[np.arange(len(labels)), labels]


def aps_scores(probs, labels):
    """Deterministic APS: cumulative sorted mass up to and including y_true."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank_of_true = np.argmax(order == labels[:, None], axis=1)
    return cum[np.arange(len(labels)), rank_of_true]


def conformal_quantile(scores, alpha):
    n = len(scores)
    q = math.ceil((n + 1) * (1 - alpha)) / n
    if q >= 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, q, method='higher'))


def lac_sets_stats(probs, labels, qhat):
    in_set = probs >= (1.0 - qhat)                       # [N, C] bool
    sizes = in_set.sum(axis=1)
    covered = in_set[np.arange(len(labels)), labels]
    return covered, sizes


def aps_sets_stats(probs, labels, qhat):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    # smallest k with cumulative mass >= qhat (always at least 1 element)
    k = (cum < qhat).sum(axis=1) + 1
    k = np.minimum(k, probs.shape[1])
    rank_of_true = np.argmax(order == labels[:, None], axis=1)
    covered = rank_of_true < k
    return covered, k


SCORE_FNS = {
    'lac': (lac_scores, lac_sets_stats),
    'aps': (aps_scores, aps_sets_stats),
}


def per_class_coverage(covered, labels, id2label):
    out = {}
    for c in np.unique(labels):
        m = labels == c
        out[id2label[int(c)]] = {
            'n': int(m.sum()),
            'coverage': float(covered[m].mean()),
        }
    return out


def evaluate_fixed(cal_probs, cal_labels, ev_probs, ev_labels, id2label):
    """Marginal CP (with per-class breakdown) + Mondrian CP, all alphas/scores."""
    results = {}
    for name, (score_fn, sets_fn) in SCORE_FNS.items():
        cal_scores = score_fn(cal_probs, cal_labels)
        results[name] = {}
        for alpha in ALPHAS:
            qhat = conformal_quantile(cal_scores, alpha)
            covered, sizes = sets_fn(ev_probs, ev_labels, qhat)
            entry = {
                'qhat': qhat,
                'coverage': float(covered.mean()),
                'avg_set_size': float(np.mean(sizes)),
                'per_class_coverage': per_class_coverage(covered, ev_labels, id2label),
            }
            # Mondrian: class-conditional quantiles from cal
            mond = {}
            m_covered = np.zeros(len(ev_labels), dtype=bool)
            m_sizes = np.zeros(len(ev_labels))
            for c in np.unique(np.concatenate([cal_labels, ev_labels])):
                cal_m = cal_labels == c
                ev_m = ev_labels == c
                if not ev_m.any():
                    continue
                if not cal_m.any():
                    mond[id2label[int(c)]] = {'n_cal': 0, 'note': 'no calibration tokens'}
                    m_covered[ev_m] = True  # vacuous full set
                    m_sizes[ev_m] = ev_probs.shape[1]
                    continue
                q_c = conformal_quantile(score_fn(cal_probs[cal_m], cal_labels[cal_m]), alpha)
                cov_c, siz_c = sets_fn(ev_probs[ev_m], ev_labels[ev_m], q_c)
                m_covered[ev_m] = cov_c
                m_sizes[ev_m] = siz_c
                mond[id2label[int(c)]] = {
                    'n_cal': int(cal_m.sum()), 'n_eval': int(ev_m.sum()),
                    'qhat': q_c, 'coverage': float(cov_c.mean()),
                    'avg_set_size': float(np.mean(siz_c)),
                }
            entry['mondrian'] = {
                'coverage': float(m_covered.mean()),
                'avg_set_size': float(np.mean(m_sizes)),
                'per_class': mond,
            }
            results[name][f'alpha_{alpha}'] = entry
    return results


def cross_conformal(cal_docs, ev_probs, ev_labels, k):
    """K-fold cross-conformal over calibration DOCS (temperature already fixed
    on the ES split; folds are document-level to respect exchangeability)."""
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(cal_docs))
    folds = np.array_split(idx, k)
    out = {}
    for name, (score_fn, sets_fn) in SCORE_FNS.items():
        out[name] = {}
        for alpha in ALPHAS:
            covs = []
            for f in range(k):
                keep = np.concatenate([folds[j] for j in range(k) if j != f])
                probs = np.concatenate([cal_docs[i][0] for i in keep])
                labels = np.concatenate([cal_docs[i][1] for i in keep])
                qhat = conformal_quantile(score_fn(probs, labels), alpha)
                covered, _ = sets_fn(ev_probs, ev_labels, qhat)
                covs.append(float(covered.mean()))
            out[name][f'alpha_{alpha}'] = {
                'fold_coverages': covs,
                'mean': float(np.mean(covs)),
                'std': float(np.std(covs)),
            }
    return out


# --------------------------------------------------------------------------
# Dataset flows
# --------------------------------------------------------------------------

def run_cord(model, bundle_dir, device, batch_size):
    es = CORDDataset(bundle_dir, split='val_es', stage='fit')
    cal = CORDDataset(bundle_dir, split='cal', stage='conformal')
    test = CORDDataset(bundle_dir, split='test', stage='conformal')
    id2label = es.id2label

    es_logits, es_labels = flatten(collect_doc_logits(model, es, device, batch_size))
    temperature = fit_temperature(es_logits, es_labels)

    cal_docs_raw = collect_doc_logits(model, cal, device, batch_size)
    cal_docs = [(probs_of(l, temperature), y.numpy()) for l, y in cal_docs_raw]
    cal_probs = np.concatenate([p for p, _ in cal_docs])
    cal_labels = np.concatenate([y for _, y in cal_docs])

    test_per_doc = collect_doc_logits(model, test, device, batch_size)
    test_metrics = compute_test_metrics(test_per_doc, id2label, temperature)
    test_logits, test_labels = flatten(test_per_doc)
    test_probs = probs_of(test_logits, temperature)
    test_labels = test_labels.numpy()

    fixed = evaluate_fixed(cal_probs, cal_labels, test_probs, test_labels, id2label)
    cross = cross_conformal(cal_docs, test_probs, test_labels, CROSS_CONFORMAL_K)

    red_flags = []
    for name in SCORE_FNS:
        for alpha in ALPHAS:
            key = f'alpha_{alpha}'
            gap = abs(fixed[name][key]['coverage'] - cross[name][key]['mean']) * 100
            if gap > CROSS_CONFORMAL_RED_FLAG_PP:
                red_flags.append(f"{name}/{key}: fixed vs cross-conformal gap {gap:.2f}pp")

    return {
        'temperature': temperature,
        'temperature_fit_split': 'val_es',
        'calibration_split': 'cal (50 docs, persisted)',
        'n_cal_tokens': int(len(cal_labels)),
        'n_test_tokens': int(len(test_labels)),
        'b_only_fields_excluded_from_bio_inconsistency': es.b_only_fields,
        'mondrian_note': ('CORD per-class analysis is non-O-only BY CONSTRUCTION: '
                          'CORD gold contains no O class. This replaces the '
                          'legacy_invalid "O≈5%" narrative.'),
        'test_metrics': test_metrics,
        'fixed_split': fixed,
        'cross_conformal_k5': cross,
        'cross_conformal_red_flags': red_flags,
    }


MULTISEED_DATASETS = {
    # dataset -> (Dataset class, ES split name)
    'funsd': (FUNSDDataset, 'es_holdout'),
    'wildreceipt': (WildReceiptDataset, 'val_es'),
    # SROIE pre-tagged addendum (PENDING PI SIGN-OFF; report with the upstream
    # fuzzy-alignment caveat — see PROVENANCE_sroie_pretagged_addendum_*.md).
    'sroie': (SROIEDataset, 'val_es'),
}


def run_multiseed(dataset_name, model, bundle_dir, device, batch_size):
    """FUNSD-style protocol: temperature on the ES split, then per persisted
    seed split {42,123,456}: quantiles on its calibration subset of test,
    evaluate on its evaluation subset; report mean±std."""
    cls, es_split = MULTISEED_DATASETS[dataset_name]
    es = cls(bundle_dir, split=es_split, stage='fit')
    id2label = es.id2label

    es_logits, es_labels = flatten(collect_doc_logits(model, es, device, batch_size))
    temperature = fit_temperature(es_logits, es_labels)

    # Point-prediction metrics on the FULL test split (seed-independent).
    full_test = cls(bundle_dir, split='test', stage='conformal')
    test_metrics = compute_test_metrics(
        collect_doc_logits(model, full_test, device, batch_size), id2label, temperature)

    per_seed = {}
    for seed in FUNSD_CALIB_SEEDS:
        cal_ids, ev_ids = conformal_split(bundle_dir, dataset_name, seed)
        cal = cls(bundle_dir, split='test', stage='conformal', doc_ids=cal_ids)
        ev = cls(bundle_dir, split='test', stage='conformal', doc_ids=ev_ids)
        cal_logits, cal_labels = flatten(collect_doc_logits(model, cal, device, batch_size))
        ev_logits, ev_labels = flatten(collect_doc_logits(model, ev, device, batch_size))
        per_seed[f'seed_{seed}'] = evaluate_fixed(
            probs_of(cal_logits, temperature), cal_labels.numpy(),
            probs_of(ev_logits, temperature), ev_labels.numpy(), id2label,
        )

    summary = {}
    for name in SCORE_FNS:
        summary[name] = {}
        for alpha in ALPHAS:
            key = f'alpha_{alpha}'
            covs = [per_seed[s][name][key]['coverage'] for s in per_seed]
            sizes = [per_seed[s][name][key]['avg_set_size'] for s in per_seed]
            summary[name][key] = {
                'coverage_mean': float(np.mean(covs)), 'coverage_std': float(np.std(covs)),
                'set_size_mean': float(np.mean(sizes)), 'set_size_std': float(np.std(sizes)),
            }

    return {
        'temperature': temperature,
        'temperature_fit_split': es_split,
        'calibration_splits': [f'{dataset_name}_calib_seed{s}.json' for s in FUNSD_CALIB_SEEDS],
        'test_metrics': test_metrics,
        'per_seed': per_seed,
        'mean_std_over_seeds': summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset', required=True,
                    choices=['cord', 'funsd', 'wildreceipt', 'sroie'])
    ap.add_argument('--bundle_dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    model = CaliDocModel.from_pretrained(args.model_path)
    model.to(args.device)

    if args.dataset == 'cord':
        results = run_cord(model, args.bundle_dir, args.device, args.batch_size)
    else:
        results = run_multiseed(args.dataset, model, args.bundle_dir, args.device, args.batch_size)

    payload = {
        'meta': {
            'dataset': args.dataset,
            'model_path': str(args.model_path),
            'bundle_dir': str(args.bundle_dir),
            'generated': datetime.now(timezone.utc).isoformat(),
            'alphas': list(ALPHAS),
            'scores': list(SCORE_FNS),
            'aps_variant': 'deterministic (no tie randomization)',
            'legacy_note': ('All Phase 1 numbers are legacy_invalid; this file is the '
                            'ONLY source for paper number updates.'),
        },
        'results': results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out)
    if args.dataset == 'cord' and results['cross_conformal_red_flags']:
        logger.warning("RED FLAGS (investigate before reporting): %s",
                       results['cross_conformal_red_flags'])


if __name__ == '__main__':
    main()
