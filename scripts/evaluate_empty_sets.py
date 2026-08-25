#!/usr/bin/env python3
"""
Empty Set & Structure-Aware Analysis

Addresses:
  - Reviewer Q2: empty prediction set rate
  - Reviewer weakness #7: structure-aware method fragility

Computes:
  1. Empty prediction set rates for LAC and APS methods
  2. Per-class empty set rates
  3. Correlation between empty sets and prediction errors
  4. Analysis of structure-aware method failure (NaN thresholds)

Inputs: per-token prediction JSONs + results JSONs from conformal_v2_multiseed/
Outputs: empty set rate tables
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from calidoc.data.funsd_dataset import FUNSD_LABELS
from calidoc.data.cord_dataset import CORD_LABELS

DATASET_LABELS = {
    'funsd': FUNSD_LABELS,
    'cord': CORD_LABELS,
}


def load_per_token_predictions(json_path: str):
    """Load per-token prediction JSON."""
    with open(json_path) as f:
        documents = json.load(f)
    return documents


def load_results_json(base_dir, dataset_name, seed):
    """Load results JSON for a given seed."""
    seed_dir = os.path.join(base_dir, dataset_name, f'seed_{seed}')
    results_files = [f for f in os.listdir(seed_dir) if f.startswith('results_v2')]
    if not results_files:
        return None
    results_path = os.path.join(seed_dir, results_files[0])
    with open(results_path) as f:
        return json.load(f)


def analyze_empty_sets(documents, label_list):
    """Analyze empty prediction sets for LAC and APS methods."""
    stats = {
        'lac': {
            'total': 0, 'empty': 0,
            'per_class': defaultdict(lambda: {'total': 0, 'empty': 0}),
            'empty_and_correct': 0,
            'empty_and_incorrect': 0,
        },
        'aps': {
            'total': 0, 'empty': 0,
            'per_class': defaultdict(lambda: {'total': 0, 'empty': 0}),
            'empty_and_correct': 0,
            'empty_and_incorrect': 0,
        },
    }

    for doc in documents:
        for tok in doc['tokens']:
            true_label = tok['true_label']
            is_correct = tok['pred_label'] == tok['true_label']

            for method in ['lac', 'aps']:
                set_key = f'prediction_set_{method}'
                size_key = f'set_size_{method}'
                pred_set = tok.get(set_key, [])
                set_size = tok.get(size_key, len(pred_set))

                stats[method]['total'] += 1
                stats[method]['per_class'][true_label]['total'] += 1

                is_empty = set_size == 0 or (isinstance(pred_set, list) and len(pred_set) == 0)

                if is_empty:
                    stats[method]['empty'] += 1
                    stats[method]['per_class'][true_label]['empty'] += 1
                    if is_correct:
                        stats[method]['empty_and_correct'] += 1
                    else:
                        stats[method]['empty_and_incorrect'] += 1

    return stats


def analyze_structure_aware_failure(results_json):
    """Analyze why structure-aware method produces NaN thresholds / zero coverage."""
    analysis = {}

    if results_json is None:
        return {'status': 'no results JSON'}

    sa_results = results_json.get('conformal_structure_aware', {})
    sa_struct = sa_results.get('structure_aware', {})

    if not sa_struct:
        return {'status': 'no structure_aware results'}

    for cov_key, cov_data in sa_struct.items():
        threshold = cov_data.get('threshold')
        coverage = cov_data.get('coverage', 0.0)
        is_nan = threshold is None or (isinstance(threshold, float) and np.isnan(threshold))

        analysis[cov_key] = {
            'threshold': threshold,
            'threshold_is_nan': is_nan,
            'coverage': coverage,
            'coverage_is_zero': coverage == 0.0,
            'temperature': cov_data.get('temperature'),
            'n_cal': cov_data.get('n_cal'),
            'n_test': cov_data.get('n_test'),
        }

    # Compare with LAC/APS to show structure-aware is uniquely broken
    for method in ['lac', 'aps']:
        method_results = sa_results.get(method, {})
        for cov_key, cov_data in method_results.items():
            if cov_key not in analysis:
                analysis[cov_key] = {}
            analysis[cov_key][f'{method}_coverage'] = cov_data.get('coverage', 0.0)
            analysis[cov_key][f'{method}_threshold'] = cov_data.get('threshold')

    return analysis


def analyze_error_correlation(documents):
    """Analyze correlation between empty sets and prediction errors."""
    results = {
        'lac': {'correct_empty': 0, 'correct_nonempty': 0,
                'incorrect_empty': 0, 'incorrect_nonempty': 0},
        'aps': {'correct_empty': 0, 'correct_nonempty': 0,
                'incorrect_empty': 0, 'incorrect_nonempty': 0},
    }

    for doc in documents:
        for tok in doc['tokens']:
            is_correct = tok['pred_label'] == tok['true_label']

            for method in ['lac', 'aps']:
                set_key = f'prediction_set_{method}'
                size_key = f'set_size_{method}'
                pred_set = tok.get(set_key, [])
                set_size = tok.get(size_key, len(pred_set))
                is_empty = set_size == 0 or (isinstance(pred_set, list) and len(pred_set) == 0)

                if is_correct and is_empty:
                    results[method]['correct_empty'] += 1
                elif is_correct and not is_empty:
                    results[method]['correct_nonempty'] += 1
                elif not is_correct and is_empty:
                    results[method]['incorrect_empty'] += 1
                else:
                    results[method]['incorrect_nonempty'] += 1

    # Compute rates
    for method in ['lac', 'aps']:
        r = results[method]
        total = sum(r.values())
        total_empty = r['correct_empty'] + r['incorrect_empty']
        total_nonempty = r['correct_nonempty'] + r['incorrect_nonempty']

        r['error_rate_given_empty'] = (
            r['incorrect_empty'] / total_empty if total_empty > 0 else 0.0)
        r['error_rate_given_nonempty'] = (
            r['incorrect_nonempty'] / total_nonempty if total_nonempty > 0 else 0.0)
        r['empty_rate_given_error'] = (
            r['incorrect_empty'] / (r['incorrect_empty'] + r['incorrect_nonempty'])
            if (r['incorrect_empty'] + r['incorrect_nonempty']) > 0 else 0.0)
        r['total'] = total

    return results


def run_analysis(dataset_name, base_dir, seeds):
    """Run empty set analysis for a dataset across seeds."""
    label_list = DATASET_LABELS[dataset_name]
    all_seed_results = {}

    for seed in seeds:
        json_path = os.path.join(
            base_dir, dataset_name, f'seed_{seed}',
            f'per_token_predictions_{dataset_name}.json')

        if not os.path.exists(json_path):
            print(f"  [SKIP] {json_path} not found")
            continue

        documents = load_per_token_predictions(json_path)
        results_json = load_results_json(base_dir, dataset_name, seed)

        print(f"\n  Seed {seed}: {len(documents)} documents")

        # Empty set analysis
        empty_stats = analyze_empty_sets(documents, label_list)

        # Error correlation
        error_corr = analyze_error_correlation(documents)

        # Structure-aware failure analysis
        sa_analysis = analyze_structure_aware_failure(results_json)

        seed_result = {
            'seed': seed,
            'n_documents': len(documents),
            'empty_sets': {},
            'error_correlation': {},
            'structure_aware_analysis': sa_analysis,
        }

        for method in ['lac', 'aps']:
            s = empty_stats[method]
            empty_rate = s['empty'] / s['total'] if s['total'] > 0 else 0.0

            per_class_rates = {}
            for label in label_list:
                pc = s['per_class'].get(label, {'total': 0, 'empty': 0})
                per_class_rates[label] = {
                    'total': pc['total'],
                    'empty': pc['empty'],
                    'empty_rate': pc['empty'] / pc['total'] if pc['total'] > 0 else 0.0,
                }

            seed_result['empty_sets'][method] = {
                'total': s['total'],
                'empty': s['empty'],
                'empty_rate': empty_rate,
                'empty_and_correct': s['empty_and_correct'],
                'empty_and_incorrect': s['empty_and_incorrect'],
                'per_class': per_class_rates,
            }

            seed_result['error_correlation'][method] = error_corr[method]

        all_seed_results[seed] = seed_result

    return all_seed_results


def print_table(results, dataset_name):
    """Print formatted results table."""
    label_list = DATASET_LABELS[dataset_name]
    seeds = sorted(results.keys())

    print(f"\n{'='*80}")
    print(f"EMPTY SET ANALYSIS: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Overview
    for method in ['lac', 'aps']:
        print(f"\n--- {method.upper()} ---")
        print(f"{'Metric':<35} ", end='')
        for seed in seeds:
            print(f"Seed {seed:>5}  ", end='')
        if len(seeds) > 1:
            print(f"  Mean±Std")
        else:
            print()
        print("-" * (35 + 12 * len(seeds) + (15 if len(seeds) > 1 else 0)))

        metrics = [
            ('Empty set rate', lambda r: r['empty_sets'][method]['empty_rate']),
            ('# Empty sets', lambda r: r['empty_sets'][method]['empty']),
            ('# Total tokens', lambda r: r['empty_sets'][method]['total']),
        ]

        for name, fn in metrics:
            print(f"{name:<35} ", end='')
            vals = []
            for seed in seeds:
                v = fn(results[seed])
                vals.append(v)
                if isinstance(v, float) and v <= 1.0:
                    print(f"{v*100:>6.1f}%  ", end='')
                else:
                    print(f"{v:>7}  ", end='')
            if len(seeds) > 1 and isinstance(vals[0], float) and vals[0] <= 1.0:
                print(f"  {np.mean(vals)*100:.1f}%±{np.std(vals)*100:.1f}%")
            else:
                print()

    # Error correlation
    print(f"\n{'='*80}")
    print(f"ERROR CORRELATION: {dataset_name.upper()}")
    print(f"{'='*80}")

    for method in ['lac', 'aps']:
        print(f"\n--- {method.upper()} ---")
        print(f"{'Metric':<40} ", end='')
        for seed in seeds:
            print(f"Seed {seed:>5}  ", end='')
        print()
        print("-" * (40 + 12 * len(seeds)))

        corr_metrics = [
            ('Error rate | empty set',
             lambda r: r['error_correlation'][method]['error_rate_given_empty']),
            ('Error rate | non-empty set',
             lambda r: r['error_correlation'][method]['error_rate_given_nonempty']),
            ('Empty rate | prediction error',
             lambda r: r['error_correlation'][method]['empty_rate_given_error']),
        ]

        for name, fn in corr_metrics:
            print(f"{name:<40} ", end='')
            for seed in seeds:
                v = fn(results[seed])
                print(f"{v*100:>6.1f}%  ", end='')
            print()

    # Per-class empty rates (LAC, averaged across seeds)
    print(f"\n{'='*80}")
    print(f"PER-CLASS EMPTY SET RATES (LAC): {dataset_name.upper()}")
    print(f"{'='*80}")

    print(f"{'Class':<25} {'n_tokens':>8} {'Empty Rate':>10}")
    print("-" * 48)

    for label in label_list:
        n_tok = int(np.mean([
            results[s]['empty_sets']['lac']['per_class'].get(label, {}).get('total', 0)
            for s in seeds]))
        empty_rate = np.mean([
            results[s]['empty_sets']['lac']['per_class'].get(label, {}).get('empty_rate', 0.0)
            for s in seeds])
        if n_tok > 0:
            print(f"{label:<25} {n_tok:>8} {empty_rate*100:>9.1f}%")

    # Structure-aware analysis
    print(f"\n{'='*80}")
    print(f"STRUCTURE-AWARE METHOD ANALYSIS: {dataset_name.upper()}")
    print(f"{'='*80}")

    for seed in seeds:
        sa = results[seed].get('structure_aware_analysis', {})
        if isinstance(sa, dict) and 'status' not in sa:
            print(f"\n  Seed {seed}:")
            for cov_key, info in sorted(sa.items()):
                if isinstance(info, dict) and 'threshold_is_nan' in info:
                    nan_str = "NaN" if info['threshold_is_nan'] else f"{info['threshold']:.6f}"
                    print(f"    {cov_key}: threshold={nan_str}, "
                          f"coverage={info['coverage']*100:.1f}%")
                    if 'lac_coverage' in info:
                        print(f"      (LAC: {info['lac_coverage']*100:.1f}%, "
                              f"APS: {info.get('aps_coverage', 0)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Empty Set & Structure-Aware Analysis")
    parser.add_argument('--base_dir', type=str,
                        default='results/conformal_v2_multiseed',
                        help='Base directory with per-token prediction JSONs')
    parser.add_argument('--datasets', nargs='+', default=['funsd', 'cord'],
                        choices=['funsd', 'cord'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--output_dir', type=str,
                        default='results/empty_set_analysis',
                        help='Output directory for JSON results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")

        results = run_analysis(dataset_name, args.base_dir, args.seeds)
        if results:
            print_table(results, dataset_name)
            all_results[dataset_name] = {
                str(k): v for k, v in results.items()}

    # Save results
    def _convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    out_path = os.path.join(args.output_dir, 'empty_set_analysis_results.json')
    with open(out_path, 'w') as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
