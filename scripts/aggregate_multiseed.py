#!/usr/bin/env python3
"""
Aggregate the 3-seed local lineage into the numbers the paper reports.

The single-seed local runs differ from the old cluster runs in both directions
(CORD lower, FUNSD higher), which is run-to-run variance amplified by early
stopping on very small ES splits (CORD val_es 50 docs, FUNSD es_holdout 15).
Reporting mean +- std over seeds {42,123,456} is the honest fix and is better
practice than any single run.

Reads results/<tag>/phase2_<dataset>_s<seed>.json (and the seed-42 file written
without a suffix) and emits a summary table plus a JSON companion.

Usage:
  python scripts/aggregate_multiseed.py --results_dir results/local_20260803 \
      --output results/local_20260803/multiseed_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATASETS = ('cord', 'funsd', 'wildreceipt', 'sroie')
SEEDS = (42, 123, 456)
ALPHAS = ('alpha_0.1', 'alpha_0.05', 'alpha_0.01')


def load(results_dir, ds, seed):
    for name in (f'phase2_{ds}_s{seed}.json',
                 f'phase2_{ds}.json' if seed == 42 else None):
        if name is None:
            continue
        p = results_dir / name
        if p.exists():
            return json.loads(p.read_text())['results']
    return None


def coverage_of(res, alpha):
    """CORD reports a fixed split; the others report mean over calib seeds."""
    if 'fixed_split' in res:
        return res['fixed_split']['lac'][alpha]['coverage'], \
               res['fixed_split']['lac'][alpha]['avg_set_size']
    ms = res['mean_std_over_seeds']['lac'][alpha]
    return ms['coverage_mean'], ms['set_size_mean']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    rd = Path(args.results_dir)

    out, rows = {}, []
    for ds in DATASETS:
        per_seed = {}
        for seed in SEEDS:
            res = load(rd, ds, seed)
            if res is None:
                continue
            tm = res['test_metrics']
            entry = {'f1': tm['f1'] * 100, 'precision': tm['precision'] * 100,
                     'recall': tm['recall'] * 100, 'ece': tm['ece'],
                     'mce': tm['mce'], 'temperature': res['temperature']}
            for a in ALPHAS:
                cov, size = coverage_of(res, a)
                entry[f'lac_cov_{a}'] = cov * 100
                entry[f'lac_size_{a}'] = size
            per_seed[seed] = entry
        if not per_seed:
            continue

        keys = next(iter(per_seed.values())).keys()
        agg = {}
        for k in keys:
            vals = [v[k] for v in per_seed.values()]
            agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                      'n_seeds': len(vals), 'values': [float(x) for x in vals]}
        out[ds] = {'per_seed': per_seed, 'aggregate': agg,
                   'seeds_present': sorted(per_seed)}
        rows.append((ds, agg, sorted(per_seed)))

    print(f'{"dataset":<13} {"seeds":<12} {"test F1":<16} {"ECE":<14} '
          f'{"LAC cov @90":<16} {"@95":<16} {"@99"}')
    for ds, agg, seeds in rows:
        def f(k, p=2):
            return f'{agg[k]["mean"]:.{p}f}+-{agg[k]["std"]:.{p}f}'
        print(f'{ds:<13} {str(seeds):<12} {f("f1"):<16} {f("ece"):<14} '
              f'{f("lac_cov_alpha_0.1"):<16} {f("lac_cov_alpha_0.05"):<16} '
              f'{f("lac_cov_alpha_0.01")}')

    Path(args.output).write_text(json.dumps({
        'seeds': list(SEEDS),
        'note': ('mean +- std over training seeds. CORD coverage is the fixed '
                 'cal-split value; FUNSD/WildReceipt/SROIE coverage is already '
                 'a mean over the three persisted calibration splits, so their '
                 'std here is across TRAINING seeds only.'),
        'datasets': out,
    }, indent=2))
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
