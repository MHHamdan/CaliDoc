#!/usr/bin/env python3
"""
Coverage under document perturbations, with a CLEAN calibration threshold.

The question is not whether accuracy degrades under corruption -- it does --
but whether a conformal threshold calibrated on clean documents still delivers
its promised coverage when the input distribution shifts. Exchangeability is
violated by construction here, so any coverage that survives is empirical, not
guaranteed; that is exactly what makes the measurement worth reporting.

Protocol: temperature and the conformal quantile are computed ONCE on clean
data (early-stopping split and clean calibration split respectively) and then
held fixed. Only the evaluation images are perturbed. Gold OCR tokens and boxes
are preserved, so this isolates the visual encoder rather than confounding it
with OCR degradation.

All 9 perturbations run: an earlier evaluation covered only 4 because the
perturbation module left its OpenCV handle unbound for every method that did
not call the lazy loader first (fixed in calidoc/data/perturbations.py).

Usage:
  python scripts/evaluate_perturbation_coverage.py \
      --model_path outputs/funsd_retrain_s42_local_20260803/best \
      --dataset funsd --bundle_dir ./data_bundle \
      --output results/local_20260803/perturbation_coverage_funsd.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from calidoc.data.bundle import FUNSD_CALIB_SEEDS, conformal_split
from calidoc.data.perturbations import DocumentPerturbations
from calidoc.models import CaliDocModel

from ablations_phase2 import DATASETS, collect
from evaluate_conformal_phase2 import (
    conformal_quantile, fit_temperature, lac_scores, lac_sets_stats, probs_of)

logger = logging.getLogger(__name__)

PERTURBATIONS = ['gaussian_blur', 'motion_blur', 'gaussian_noise',
                 'salt_pepper_noise', 'jpeg_compression', 'brightness',
                 'contrast', 'ink_bleed', 'paper_texture']
SEVERITIES = [1, 2, 3]
ALPHA = 0.10


class FixedPerturbation:
    """Drop-in replacement for the dataset's augmentor slot that applies ONE
    named perturbation at ONE severity, instead of sampling randomly.

    Must be a module-level class holding only picklable state: the dataloader
    uses worker processes, so the dataset (and therefore this object) is
    pickled. DocumentPerturbations is constructed lazily inside the worker for
    the same reason.
    """

    def __init__(self, name, severity):
        self.name = name
        self.severity = severity
        self._pert = None

    def apply_random_perturbation(self, image_np, severity=None):
        if self._pert is None:
            self._pert = DocumentPerturbations()
        fn = getattr(self._pert, self.name)
        return fn(image_np, severity=self.severity), self.name, self.severity


def perturbed(dataset, name, severity):
    """Context-free helper: point the dataset at a fixed perturbation.
    Returns a restore callable."""
    prev_flag = dataset.apply_augmentation
    prev_aug = getattr(dataset, 'augmentor', None)
    dataset.apply_augmentation = True
    dataset.augmentor = FixedPerturbation(name, severity)

    def restore():
        dataset.apply_augmentation = prev_flag
        dataset.augmentor = prev_aug
    return restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--bundle_dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cls, es_split = DATASETS[args.dataset]
    bundle_dir = Path(args.bundle_dir)
    device = torch.device(args.device)

    mp = Path(args.model_path)
    cfg = json.loads((mp / 'config.json').read_text())
    model = CaliDocModel(encoder_type=cfg['encoder_type'],
                         encoder_config=cfg['encoder_config'],
                         use_quality_routing=False, use_calibration=False)
    model.load_state_dict(torch.load(mp / 'pytorch_model.bin', map_location='cpu',
                                     weights_only=False))
    model.to(device)

    # ---- clean calibration, held fixed ----
    es = cls(bundle_dir, split=es_split, stage='fit')
    id2label = es.id2label
    es_pd = collect(model, es, device, args.batch_size)
    temperature = fit_temperature(torch.cat([l for l, _, _ in es_pd]),
                                  torch.cat([y for _, y, _ in es_pd]))

    if args.dataset == 'cord':
        cal_ds = cls(bundle_dir, split='cal', stage='conformal')
        ev_ds = cls(bundle_dir, split='test', stage='conformal')
    else:
        cal_ids, ev_ids = conformal_split(bundle_dir, args.dataset,
                                          FUNSD_CALIB_SEEDS[0])
        cal_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=cal_ids)
        ev_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=ev_ids)

    cal_pd = collect(model, cal_ds, device, args.batch_size)
    cal_probs = np.concatenate([probs_of(l, temperature) for l, _, _ in cal_pd])
    cal_labels = np.concatenate([y.numpy() for _, y, _ in cal_pd])
    qhat = conformal_quantile(lac_scores(cal_probs, cal_labels), ALPHA)
    logger.info('clean qhat @alpha=%.2f: %.6f (T=%.4f)', ALPHA, qhat, temperature)

    from seqeval.metrics import f1_score

    def evaluate(dataset, label):
        pd_ = collect(model, dataset, device, args.batch_size)
        probs = np.concatenate([probs_of(l, temperature) for l, _, _ in pd_])
        labels = np.concatenate([y.numpy() for _, y, _ in pd_])
        covered, sizes = lac_sets_stats(probs, labels, qhat)
        true_seqs, pred_seqs = [], []
        for lg, y, _ in pd_:
            pr = probs_of(lg, temperature).argmax(axis=1)
            yy = y.numpy()
            true_seqs.append([id2label[int(t)] for t in yy])
            pred_seqs.append([id2label[int(p)] for p in pr])
        return {
            'coverage': float(covered.mean()),
            'avg_set_size': float(np.mean(sizes)),
            'f1': float(f1_score(true_seqs, pred_seqs)),
            'token_accuracy': float((probs.argmax(1) == labels).mean()),
            'n_tokens': int(len(labels)),
        }

    clean = evaluate(ev_ds, 'clean')
    logger.info('clean: cov %.4f  F1 %.4f', clean['coverage'], clean['f1'])

    results = {}
    for name in PERTURBATIONS:
        results[name] = {}
        for sev in SEVERITIES:
            restore = perturbed(ev_ds, name, sev)
            try:
                r = evaluate(ev_ds, f'{name}/{sev}')
                r['coverage_drop_pp'] = (clean['coverage'] - r['coverage']) * 100
                r['f1_drop_pp'] = (clean['f1'] - r['f1']) * 100
                results[name][f'severity_{sev}'] = r
                logger.info('%-18s sev %d: cov %.4f (%+.2fpp)  F1 %.4f (%+.2fpp)',
                            name, sev, r['coverage'], -r['coverage_drop_pp'],
                            r['f1'], -r['f1_drop_pp'])
            except Exception as e:                               # noqa: BLE001
                logger.warning('%s sev %d FAILED: %s', name, sev, e)
                results[name][f'severity_{sev}'] = {'error': str(e)}
            finally:
                restore()

    drops = [v['coverage_drop_pp'] for p in results.values() for v in p.values()
             if 'coverage_drop_pp' in v]
    f1drops = [v['f1_drop_pp'] for p in results.values() for v in p.values()
               if 'f1_drop_pp' in v]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'dataset': args.dataset,
        'model_path': str(mp),
        'protocol': {
            'alpha': ALPHA,
            'temperature': temperature,
            'clean_qhat': qhat,
            'note': ('Temperature and conformal quantile are fitted on CLEAN '
                     'data and held fixed; only evaluation images are '
                     'perturbed. Gold OCR tokens and boxes are preserved, so '
                     'this isolates the visual encoder. Exchangeability is '
                     'violated by construction, so surviving coverage is '
                     'empirical rather than guaranteed.'),
            'n_perturbations': len(PERTURBATIONS),
            'severities': SEVERITIES,
        },
        'clean': clean,
        'perturbations': results,
        'summary': {
            'mean_coverage_drop_pp': float(np.mean(drops)) if drops else None,
            'max_coverage_drop_pp': float(np.max(drops)) if drops else None,
            'mean_f1_drop_pp': float(np.mean(f1drops)) if f1drops else None,
            'max_f1_drop_pp': float(np.max(f1drops)) if f1drops else None,
            'n_cells': len(drops),
        },
    }, indent=2))
    logger.info('Wrote %s | mean coverage drop %.2fpp, max %.2fpp over %d cells',
                out, np.mean(drops) if drops else float('nan'),
                np.max(drops) if drops else float('nan'), len(drops))


if __name__ == '__main__':
    main()
