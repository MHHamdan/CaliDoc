#!/usr/bin/env python3
"""
Cache temperature-scaled probabilities once per checkpoint so that every
downstream post-hoc analysis (conditional-CP baselines, O-class/span analysis,
label-error-detector benchmarks) is CPU-only and instant.

Writes results/cache/<dataset>_<tag>.npz containing, for each protocol unit:

    <unit>_probs      [N, C] float32  temperature-scaled softmax
    <unit>_labels     [N]    int64    gold label ids
    <unit>_ent        [N]    float32  normalised layout-attention entropy
    <unit>_docstart   [D+1]  int64    document boundaries into the above

plus `temperature`, `id2label`, and the unit list in a JSON sidecar.

Units mirror scripts/evaluate_conformal_phase2.py exactly:
  CORD          -> cal (50 persisted docs) + test
  FUNSD/WR/SROIE-> per persisted seed {42,123,456}: cal / eval subsets of test,
                   plus the full test split.

Usage:
  python scripts/cache_probs.py --model_path outputs/<exp>/best \
      --dataset cord --bundle_dir ./data_bundle --tag local_20260803
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
from calidoc.models import CaliDocModel

from ablations_phase2 import DATASETS, collect
from evaluate_conformal_phase2 import fit_temperature, probs_of

logger = logging.getLogger(__name__)


def pack(per_doc, temperature):
    probs = np.concatenate([probs_of(l, temperature) for l, _, _ in per_doc]).astype(np.float32)
    labels = np.concatenate([y.numpy() for _, y, _ in per_doc]).astype(np.int64)
    ents = [e for _, _, e in per_doc]
    ent = (np.concatenate(ents).astype(np.float32)
           if all(e is not None for e in ents) else np.zeros(len(labels), np.float32))
    lens = [len(y) for _, y, _ in per_doc]
    docstart = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return probs, labels, ent, docstart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--bundle_dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out_dir', default='results/cache')
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
    model = CaliDocModel(
        encoder_type=cfg['encoder_type'], encoder_config=cfg['encoder_config'],
        use_quality_routing=cfg.get('use_quality_routing', False),
        use_calibration=cfg.get('use_calibration', False))
    model.load_state_dict(torch.load(mp / 'pytorch_model.bin', map_location='cpu',
                                     weights_only=False))
    model.to(device)

    es = cls(bundle_dir, split=es_split, stage='fit')
    id2label = es.id2label
    es_pd = collect(model, es, device, args.batch_size)
    temperature = fit_temperature(torch.cat([l for l, _, _ in es_pd]),
                                  torch.cat([y for _, y, _ in es_pd]))

    arrays, units = {}, []

    def add(unit, per_doc):
        p, y, e, d = pack(per_doc, temperature)
        arrays[f'{unit}_probs'] = p
        arrays[f'{unit}_labels'] = y
        arrays[f'{unit}_ent'] = e
        arrays[f'{unit}_docstart'] = d
        units.append(unit)
        logger.info("unit %s: %d docs, %d tokens", unit, len(d) - 1, len(y))

    full_test = cls(bundle_dir, split='test', stage='conformal')
    add('test', collect(model, full_test, device, args.batch_size, want_attention=True))

    if args.dataset in ('cord', 'wildreceipt_traincal'):
        add('cal', collect(model, cls(bundle_dir, split='cal', stage='conformal'),
                           device, args.batch_size, want_attention=True))
    else:
        for seed in FUNSD_CALIB_SEEDS:
            cal_ids, ev_ids = conformal_split(bundle_dir, args.dataset, seed)
            cal_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=cal_ids)
            ev_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=ev_ids)
            add(f'cal_s{seed}',
                collect(model, cal_ds, device, args.batch_size, want_attention=True))
            add(f'ev_s{seed}',
                collect(model, ev_ds, device, args.batch_size, want_attention=True))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f'{args.dataset}_{args.tag}.npz'
    np.savez_compressed(npz, **arrays)
    (out_dir / f'{args.dataset}_{args.tag}.json').write_text(json.dumps({
        'dataset': args.dataset,
        'model_path': str(mp),
        'temperature': temperature,
        'temperature_fit_split': es_split,
        'units': units,
        'id2label': {str(k): v for k, v in id2label.items()},
        'num_classes': len(id2label),
    }, indent=2))
    logger.info("Wrote %s", npz)


if __name__ == '__main__':
    main()
