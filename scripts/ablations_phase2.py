#!/usr/bin/env python3
"""
Phase 2 ablation suite -- the producer of results/ablations_<dataset>.json.

Regenerates everything consolidated in the supplementary "Pending Re-Run"
section, on the corrected Phase 2 checkpoints, under the SAME clean protocol
enforced by calidoc.data.bundle:

    temperature fitted ONLY on the ES split (val_es / es_holdout)
    -> conformal quantiles ONLY on calibration docs
    -> everything reported on held-out evaluation docs.

Split flows mirror scripts/evaluate_conformal_phase2.py exactly (that script
stays the sole producer of phase2_results.json; this one never touches it):
  CORD        : one unit  (cal=the 50 persisted cal docs, eval=test)
  FUNSD/WR/   : one unit per persisted seed {42,123,456}, each splitting test
  SROIE         into a calibration and an evaluation subset; mean+-std reported.

Ablations produced
  selective_prediction  risk-coverage curve + AURC (token-level)
  scoring_ablation      LAC vs deterministic APS vs structure-aware
                        (LAC + lambda_layout * normalised layout-attention
                        entropy; the OCR term is identically zero because no
                        bundle dataset carries OCR confidence -- the paper's
                        lambda_O = 0 caveat)
  raps                  deterministic RAPS over a (k_reg, lambda) grid
  empty_sets            fraction of empty LAC sets per alpha
  entity_level_cp       BIO span-level coverage (a span counts as covered only
                        if EVERY one of its tokens' sets contains its gold tag)
  per_class_f1          seqeval per-entity-type P/R/F1
  temperature_ece       ECE/MCE at T=1 vs fitted T + reliability-bin data
  mc_dropout            N stochastic passes: predictive entropy + CP on the
                        MC-averaged predictive distribution

Determinism note carried into the JSON metadata: APS and RAPS are the
DETERMINISTIC variants (no tie-breaking randomisation), matching
evaluate_conformal_phase2.py. This is why they over-cover on large label
spaces -- reported, not silently fixed.

Usage:
  python scripts/ablations_phase2.py \
      --model_path outputs/cord_retrain_s42_local_20260803/best \
      --dataset cord --bundle_dir ./data_bundle \
      --output results/ablations_cord.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
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
from calidoc.data.sroie_dataset import SROIEDataset
from calidoc.data.wildreceipt_dataset import WildReceiptDataset, WildReceiptTraincalDataset
from calidoc.models import CaliDocModel

# Reuse the audited primitives; never re-implement the protocol.
from evaluate_conformal_phase2 import (
    ALPHAS,
    aps_scores,
    aps_sets_stats,
    conformal_quantile,
    fit_temperature,
    lac_scores,
    lac_sets_stats,
    probs_of,
)

logger = logging.getLogger(__name__)

DATASETS = {
    'cord': (CORDDataset, 'val_es'),
    'funsd': (FUNSDDataset, 'es_holdout'),
    'wildreceipt': (WildReceiptDataset, 'val_es'),
    'wildreceipt_traincal': (WildReceiptTraincalDataset, 'val_es'),
    'sroie': (SROIEDataset, 'val_es'),
}

LAMBDA_LAYOUT_GRID = (0.0, 0.05, 0.1, 0.2)
RAPS_KREG_GRID = (1, 2, 5)
RAPS_LAMBDA_GRID = (0.001, 0.01, 0.1)
MC_DROPOUT_PASSES = 20
RELIABILITY_BINS = 15


# --------------------------------------------------------------------------
# Collection: logits, labels, and per-token layout-attention entropy
# --------------------------------------------------------------------------

def _attention_entropy(attentions, n_text):
    """Normalised entropy of each text token's last-layer self-attention
    distribution, averaged over heads.

    LayoutLMv3 concatenates image patches after the text tokens, so the
    attention matrices are [B, H, L, L] with L = n_text + n_patches. Keys span
    the full sequence (a text token attending to layout/image patches is
    exactly the structure signal we want), but QUERIES are restricted to the
    n_text positions that carry labels."""
    last = attentions[-1]                       # [B, H, L, L]
    a = last.float().mean(dim=1)[:, :n_text, :]  # heads averaged, text queries
    a = a / (a.sum(dim=-1, keepdim=True) + 1e-10)
    ent = -(a * torch.log(a + 1e-10)).sum(dim=-1)          # [B, n_text]
    return (ent / np.log(a.shape[-1])).cpu()


def collect(model, dataset, device, batch_size, want_attention=False):
    """Per-doc (logits, labels, attn_entropy) for tokens with label != -100."""
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    per_doc = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            out = model(
                images=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                bbox=batch['bbox'],
                labels=None,
                return_loss=False,
            )
            logits = out['logits'].float().cpu()
            labels = batch['labels'].cpu()
            ent = None
            if want_attention and out.get('attentions') is not None:
                ent = _attention_entropy(out['attentions'], labels.shape[1])
            for i in range(logits.shape[0]):
                m = labels[i] != -100
                e = ent[i][m].numpy() if ent is not None else None
                per_doc.append((logits[i][m], labels[i][m], e))
    assert len(per_doc) == len(dataset), (len(per_doc), len(dataset))
    return per_doc


def _cat(per_doc, temperature):
    probs = np.concatenate([probs_of(l, temperature) for l, _, _ in per_doc])
    labels = np.concatenate([y.numpy() for _, y, _ in per_doc])
    ents = [e for _, _, e in per_doc]
    ent = np.concatenate(ents) if all(e is not None for e in ents) else None
    return probs, labels, ent


# --------------------------------------------------------------------------
# Ablation 1: selective prediction (risk-coverage)
# --------------------------------------------------------------------------

def selective_prediction(probs, labels, n_points=101):
    """Abstain on the least-confident tokens; report token error rate on the
    retained fraction. AURC = area under the risk-coverage curve."""
    conf = probs.max(axis=1)
    err = (probs.argmax(axis=1) != labels).astype(float)
    order = np.argsort(-conf)                    # most confident first
    err_sorted = err[order]
    cum_err = np.cumsum(err_sorted)
    n = len(err_sorted)

    cov_grid, risks = [], []
    for cov in np.linspace(1.0 / n, 1.0, n_points):
        k = max(1, int(round(cov * n)))
        cov_grid.append(k / n)
        risks.append(cum_err[k - 1] / k)
    aurc = float(np.trapezoid(risks, cov_grid)) if hasattr(np, 'trapezoid') \
        else float(np.trapz(risks, cov_grid))
    return {
        'coverage': [float(c) for c in cov_grid],
        'risk': [float(r) for r in risks],
        'aurc': aurc,
        'full_coverage_risk': float(err.mean()),
        'risk_at_coverage': {
            f'{c:.2f}': float(cum_err[max(1, int(round(c * n))) - 1]
                              / max(1, int(round(c * n))))
            for c in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
        },
    }


# --------------------------------------------------------------------------
# Ablation 2: scoring ablation (LAC vs APS vs structure-aware)
# --------------------------------------------------------------------------

def structure_aware_scores(probs, labels, ent, lam):
    """LAC + lambda_layout * normalised layout-attention entropy.
    The OCR term is omitted: no bundle dataset carries OCR confidence."""
    return lac_scores(probs, labels) + lam * ent


def structure_aware_sets(probs, ent, qhat, lam):
    """Set = {y : (1 - p_y) + lam * entropy_token <= qhat}. The entropy term is
    per-token, so it shifts every class score in that token equally."""
    thresh = (qhat - lam * ent)[:, None]
    in_set = (1.0 - probs) <= thresh
    return in_set


def scoring_ablation(cal, ev):
    cal_probs, cal_labels, cal_ent = cal
    ev_probs, ev_labels, ev_ent = ev
    out = {}

    for name, score_fn, sets_fn in (
        ('lac', lac_scores, lac_sets_stats),
        ('aps', aps_scores, aps_sets_stats),
    ):
        cal_s = score_fn(cal_probs, cal_labels)
        out[name] = {}
        for alpha in ALPHAS:
            qhat = conformal_quantile(cal_s, alpha)
            covered, sizes = sets_fn(ev_probs, ev_labels, qhat)
            out[name][f'alpha_{alpha}'] = {
                'qhat': qhat,
                'coverage': float(covered.mean()),
                'avg_set_size': float(np.mean(sizes)),
            }

    if cal_ent is None or ev_ent is None:
        out['structure_aware'] = {'status': 'unavailable (no attentions returned)'}
        return out

    out['structure_aware'] = {}
    for lam in LAMBDA_LAYOUT_GRID:
        cal_s = structure_aware_scores(cal_probs, cal_labels, cal_ent, lam)
        entry = {}
        for alpha in ALPHAS:
            qhat = conformal_quantile(cal_s, alpha)
            in_set = structure_aware_sets(ev_probs, ev_ent, qhat, lam)
            sizes = in_set.sum(axis=1)
            covered = in_set[np.arange(len(ev_labels)), ev_labels]
            entry[f'alpha_{alpha}'] = {
                'qhat': qhat,
                'coverage': float(covered.mean()),
                'avg_set_size': float(np.mean(sizes)),
                'empty_set_rate': float((sizes == 0).mean()),
            }
        out['structure_aware'][f'lambda_layout_{lam}'] = entry
    out['structure_aware']['_note'] = (
        'lambda_layout=0.0 is identical to LAC by construction and is included '
        'as the internal control. lambda_ocr is fixed at 0: no bundle dataset '
        'carries OCR confidence.')
    return out


# --------------------------------------------------------------------------
# Ablation 3: deterministic RAPS
# --------------------------------------------------------------------------

def raps_scores(probs, labels, k_reg, lam):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank_of_true = np.argmax(order == labels[:, None], axis=1)      # 0-based
    base = cum[np.arange(len(labels)), rank_of_true]
    penalty = lam * np.maximum(0, (rank_of_true + 1) - k_reg)
    return base + penalty


def raps_sets_stats(probs, labels, qhat, k_reg, lam):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    ranks = np.arange(1, probs.shape[1] + 1)[None, :]
    score_at_rank = cum + lam * np.maximum(0, ranks - k_reg)
    keep = score_at_rank <= qhat
    keep[:, 0] = True                          # always keep the top class
    k = keep.sum(axis=1)
    rank_of_true = np.argmax(order == labels[:, None], axis=1)
    return rank_of_true < k, k


def raps_ablation(cal, ev):
    cal_probs, cal_labels, _ = cal
    ev_probs, ev_labels, _ = ev
    out = {}
    for k_reg in RAPS_KREG_GRID:
        for lam in RAPS_LAMBDA_GRID:
            cal_s = raps_scores(cal_probs, cal_labels, k_reg, lam)
            entry = {}
            for alpha in ALPHAS:
                qhat = conformal_quantile(cal_s, alpha)
                covered, sizes = raps_sets_stats(ev_probs, ev_labels, qhat, k_reg, lam)
                entry[f'alpha_{alpha}'] = {
                    'qhat': qhat,
                    'coverage': float(covered.mean()),
                    'avg_set_size': float(np.mean(sizes)),
                }
            out[f'k_reg_{k_reg}_lambda_{lam}'] = entry
    return out


# --------------------------------------------------------------------------
# Ablation 4: empty sets (LAC only; APS/RAPS keep >=1 class by construction)
# --------------------------------------------------------------------------

def empty_set_analysis(cal, ev, id2label):
    cal_probs, cal_labels, _ = cal
    ev_probs, ev_labels, _ = ev
    cal_s = lac_scores(cal_probs, cal_labels)
    out = {}
    for alpha in ALPHAS:
        qhat = conformal_quantile(cal_s, alpha)
        in_set = ev_probs >= (1.0 - qhat)
        sizes = in_set.sum(axis=1)
        empty = sizes == 0
        by_class = defaultdict(lambda: [0, 0])
        for c in np.unique(ev_labels):
            m = ev_labels == c
            by_class[id2label[int(c)]] = [int(empty[m].sum()), int(m.sum())]
        out[f'alpha_{alpha}'] = {
            'qhat': qhat,
            'empty_set_rate': float(empty.mean()),
            'n_empty': int(empty.sum()),
            'n_tokens': int(len(sizes)),
            'accuracy_on_empty': (float((ev_probs[empty].argmax(axis=1)
                                         == ev_labels[empty]).mean())
                                  if empty.any() else None),
            'empty_by_class': {k: {'n_empty': v[0], 'n': v[1]}
                               for k, v in by_class.items() if v[0] > 0},
        }
    return out


# --------------------------------------------------------------------------
# Ablation 5: entity-level (BIO span) conformal coverage
# --------------------------------------------------------------------------

def _gold_spans(tags):
    """BIO spans from a gold tag sequence -> list of (start, end_exclusive)."""
    spans, start = [], None
    for i, t in enumerate(tags):
        if t.startswith('B-'):
            if start is not None:
                spans.append((start, i))
            start = i
        elif t.startswith('I-'):
            if start is None:
                start = i                      # tolerate a leading I- (noisy gold)
        else:                                   # 'O'
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def entity_level_cp(cal, ev_per_doc, id2label, temperature):
    """A gold entity is covered iff every token in the span has its gold tag in
    that token's prediction set. Token-level CP gives no span-level guarantee;
    this quantifies the gap."""
    cal_probs, cal_labels, _ = cal
    cal_s = lac_scores(cal_probs, cal_labels)
    out = {}
    for alpha in ALPHAS:
        qhat = conformal_quantile(cal_s, alpha)
        n_spans = n_cov = 0
        tok_cov = tok_n = 0
        size_acc = []
        for logits, labels, _ in ev_per_doc:
            probs = probs_of(logits, temperature)
            y = labels.numpy()
            in_set = probs >= (1.0 - qhat)
            covered = in_set[np.arange(len(y)), y]
            tok_cov += int(covered.sum()); tok_n += len(y)
            tags = [id2label[int(t)] for t in y]
            for s, e in _gold_spans(tags):
                n_spans += 1
                if covered[s:e].all():
                    n_cov += 1
                size_acc.append(float(in_set[s:e].sum(axis=1).max()))
        out[f'alpha_{alpha}'] = {
            'qhat': qhat,
            'entity_coverage': (n_cov / n_spans) if n_spans else None,
            'n_entities': n_spans,
            'token_coverage': (tok_cov / tok_n) if tok_n else None,
            'avg_max_set_size_in_span': float(np.mean(size_acc)) if size_acc else None,
        }
    return out


# --------------------------------------------------------------------------
# Ablation 6: per-class F1
# --------------------------------------------------------------------------

def per_class_f1(per_doc, id2label, temperature):
    from seqeval.metrics import classification_report
    true_seqs, pred_seqs = [], []
    for logits, labels, _ in per_doc:
        preds = probs_of(logits, temperature).argmax(axis=1)
        y = labels.numpy()
        true_seqs.append([id2label[int(t)] for t in y])
        pred_seqs.append([id2label[int(p)] for p in preds])
    rep = classification_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
    return {k: {m: float(v) for m, v in d.items()} if isinstance(d, dict) else float(d)
            for k, d in rep.items()}


# --------------------------------------------------------------------------
# Ablation 7: temperature-scaling ECE deltas + reliability bins
# --------------------------------------------------------------------------

def _ece_bins(probs, labels, n_bins=RELIABILITY_BINS):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            bins.append({'bin_lower': float(lo), 'bin_upper': float(hi),
                         'n': 0, 'confidence': None, 'accuracy': None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        gap = abs(a - c)
        ece += float(m.mean()) * gap
        mce = max(mce, gap)
        bins.append({'bin_lower': float(lo), 'bin_upper': float(hi),
                     'n': int(m.sum()), 'confidence': c, 'accuracy': a})
    return {'ece': ece * 100, 'mce': mce * 100, 'bins': bins}


def temperature_ece(per_doc, temperature):
    raw = np.concatenate([probs_of(l, 1.0) for l, _, _ in per_doc])
    scaled = np.concatenate([probs_of(l, temperature) for l, _, _ in per_doc])
    labels = np.concatenate([y.numpy() for _, y, _ in per_doc])
    before, after = _ece_bins(raw, labels), _ece_bins(scaled, labels)
    return {
        'temperature': temperature,
        'uncalibrated': before,
        'temperature_scaled': after,
        'delta_ece': after['ece'] - before['ece'],
        'delta_mce': after['mce'] - before['mce'],
    }


# --------------------------------------------------------------------------
# Ablation 8: MC dropout
# --------------------------------------------------------------------------

def mc_dropout(model, dataset, device, batch_size, cal, temperature, n_passes):
    """Enable dropout at inference; average the predictive distribution over
    n_passes. Reports predictive entropy and CP on the MC-averaged probs using
    the SAME clean calibration quantile."""
    def enable_dropout(m):
        for mod in m.modules():
            if mod.__class__.__name__.startswith('Dropout'):
                mod.train()

    model.eval()
    enable_dropout(model)
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    acc = None
    labels_all = None
    torch.manual_seed(42)
    with torch.no_grad():
        for p in range(n_passes):
            chunks, lab_chunks = [], []
            for batch in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                out = model(images=batch['pixel_values'], input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], bbox=batch['bbox'],
                            labels=None, return_loss=False)
                lg = out['logits'].float().cpu()
                lb = batch['labels'].cpu()
                for i in range(lg.shape[0]):
                    m = lb[i] != -100
                    chunks.append(F.softmax(lg[i][m] / temperature, dim=-1).numpy())
                    if p == 0:
                        lab_chunks.append(lb[i][m].numpy())
            probs = np.concatenate(chunks)
            acc = probs if acc is None else acc + probs
            if p == 0:
                labels_all = np.concatenate(lab_chunks)
    model.eval()
    mc_probs = acc / n_passes

    ent = -(mc_probs * np.log(mc_probs + 1e-12)).sum(axis=1)
    cal_probs, cal_labels, _ = cal
    cal_s = lac_scores(cal_probs, cal_labels)
    cp = {}
    for alpha in ALPHAS:
        qhat = conformal_quantile(cal_s, alpha)
        covered, sizes = lac_sets_stats(mc_probs, labels_all, qhat)
        cp[f'alpha_{alpha}'] = {'coverage': float(covered.mean()),
                                'avg_set_size': float(np.mean(sizes))}
    return {
        'n_passes': n_passes,
        'accuracy': float((mc_probs.argmax(axis=1) == labels_all).mean()),
        'mean_predictive_entropy': float(ent.mean()),
        'ece': _ece_bins(mc_probs, labels_all)['ece'],
        'conformal_on_mc_probs': cp,
        'note': ('Calibration quantile is taken from the deterministic (non-MC) '
                 'calibration pass, so this measures the effect of MC averaging '
                 'on the evaluation distribution only.'),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def build_units(dataset_name, cls, bundle_dir, model, device, bs):
    """Return (temperature, id2label, full_test_per_doc, [(name, cal, ev, ev_per_doc)])."""
    es_split = DATASETS[dataset_name][1]
    es = cls(bundle_dir, split=es_split, stage='fit')
    id2label = es.id2label

    es_pd = collect(model, es, device, bs)
    es_logits = torch.cat([l for l, _, _ in es_pd])
    es_labels = torch.cat([y for _, y, _ in es_pd])
    temperature = fit_temperature(es_logits, es_labels)

    full_test = cls(bundle_dir, split='test', stage='conformal')
    full_test_pd = collect(model, full_test, device, bs, want_attention=True)

    units = []
    if dataset_name == 'cord':
        cal_ds = cls(bundle_dir, split='cal', stage='conformal')
        cal_pd = collect(model, cal_ds, device, bs, want_attention=True)
        units.append(('fixed_split',
                      _cat(cal_pd, temperature), _cat(full_test_pd, temperature),
                      full_test_pd))
    else:
        for seed in FUNSD_CALIB_SEEDS:
            cal_ids, ev_ids = conformal_split(bundle_dir, dataset_name, seed)
            cal_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=cal_ids)
            ev_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=ev_ids)
            cal_pd = collect(model, cal_ds, device, bs, want_attention=True)
            ev_pd = collect(model, ev_ds, device, bs, want_attention=True)
            units.append((f'seed_{seed}',
                          _cat(cal_pd, temperature), _cat(ev_pd, temperature), ev_pd))
    return temperature, id2label, full_test_pd, units


def aggregate(per_unit):
    """mean+-std of every leaf float across units, preserving structure.

    Only keys present in EVERY unit are aggregated: per-class dictionaries are
    sparse (a rare class can be absent from one seed's evaluation subset), and
    averaging over the units that happen to contain a class would silently
    change the denominator per class."""
    per_unit = [u for u in per_unit if isinstance(u, dict)]
    if len(per_unit) <= 1:
        return None
    keys = set(per_unit[0])
    for u in per_unit[1:]:
        keys &= set(u)
    out = {}
    for k in sorted(keys):
        vals = [u[k] for u in per_unit]
        if all(isinstance(v, dict) for v in vals):
            sub = aggregate(vals)
            if sub:
                out[k] = sub
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool)
                 for v in vals):
            out[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                      'n_units': len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--bundle_dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--mc_passes', type=int, default=MC_DROPOUT_PASSES)
    ap.add_argument('--skip_mc_dropout', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    bundle_dir = Path(args.bundle_dir)
    cls, _ = DATASETS[args.dataset]
    device = torch.device(args.device)

    model_path = Path(args.model_path)
    cfg = json.loads((model_path / 'config.json').read_text())
    model = CaliDocModel(
        encoder_type=cfg['encoder_type'],
        encoder_config=cfg['encoder_config'],
        use_quality_routing=cfg.get('use_quality_routing', False),
        use_calibration=cfg.get('use_calibration', False),
    )
    state = torch.load(model_path / 'pytorch_model.bin', map_location='cpu',
                       weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    logger.info("Loaded %s", model_path)

    temperature, id2label, full_test_pd, units = build_units(
        args.dataset, cls, bundle_dir, model, device, args.batch_size)

    per_unit = []
    for name, cal, ev, ev_pd in units:
        logger.info("Ablations on unit %s", name)
        per_unit.append({
            'unit': name,
            'selective_prediction': selective_prediction(ev[0], ev[1]),
            'scoring_ablation': scoring_ablation(cal, ev),
            'raps': raps_ablation(cal, ev),
            'empty_sets': empty_set_analysis(cal, ev, id2label),
            'entity_level_cp': entity_level_cp(cal, ev_pd, id2label, temperature),
        })

    results = {
        'dataset': args.dataset,
        'model_path': str(model_path),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'protocol': {
            'temperature_fit_split': DATASETS[args.dataset][1],
            'temperature': temperature,
            'alphas': list(ALPHAS),
            'units': [u[0] for u in units],
            'determinism_note': ('APS and RAPS are the DETERMINISTIC variants '
                                 '(no tie-breaking randomisation), matching '
                                 'evaluate_conformal_phase2.py.'),
            'ocr_confidence': ('absent from every bundle dataset; the '
                               'structure-aware score therefore uses '
                               'lambda_ocr = 0 (paper caveat).'),
        },
        'per_unit': per_unit,
        'per_class_f1': per_class_f1(full_test_pd, id2label, temperature),
        'temperature_ece': temperature_ece(full_test_pd, temperature),
    }
    agg = aggregate([{k: v for k, v in u.items() if k != 'unit'} for u in per_unit])
    if agg:
        results['mean_std_over_units'] = agg

    if not args.skip_mc_dropout:
        logger.info("MC dropout (%d passes)", args.mc_passes)
        test_ds = cls(bundle_dir, split='test', stage='conformal')
        results['mc_dropout'] = mc_dropout(
            model, test_ds, device, args.batch_size,
            units[0][1], temperature, args.mc_passes)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out)


if __name__ == '__main__':
    main()
