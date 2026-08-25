#!/usr/bin/env python3
"""
Cross-Conformal Validation Evaluation

Addresses:
  - Reviewer weakness #3: CORD val=cal protocol overlap concern
  - Reviewer Q3: calibration/test data contamination

Computes:
  K=5 document-level cross-conformal prediction:
  1. Split documents into K folds (document-level, not token-level)
  2. For each fold: calibrate on other K-1 folds, evaluate on held-out fold
  3. Pool coverage across folds for overall cross-conformal coverage
  4. Compare with fixed-split results from the original pipeline

Inputs: per-token prediction JSONs from results/conformal_v2_multiseed/
Outputs: cross-conformal coverage ± std vs existing fixed-split results
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


def compute_nonconformity_scores(tokens):
    """Compute LAC nonconformity scores: 1 - softmax(true_class)."""
    scores = []
    for tok in tokens:
        if tok['pred_label'] == tok['true_label']:
            scores.append(1.0 - tok['confidence'])
        else:
            scores.append(1.0)
    return np.array(scores)


def compute_threshold(scores, alpha=0.10):
    """Compute conformal threshold at level 1-alpha."""
    n = len(scores)
    if n == 0:
        return float('inf')
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method='higher'))


def evaluate_fold_coverage(test_tokens, test_scores, threshold):
    """Evaluate coverage on a test fold."""
    covered = test_scores <= threshold
    return {
        'coverage': float(np.mean(covered)) if len(covered) > 0 else 0.0,
        'n_tokens': len(test_tokens),
        'n_covered': int(np.sum(covered)),
    }


def create_document_folds(documents, k=5, seed=42):
    """Create K document-level folds.

    Returns list of K lists, each containing document indices.
    """
    np.random.seed(seed)
    n_docs = len(documents)
    perm = np.random.permutation(n_docs)
    folds = [[] for _ in range(k)]
    for i, doc_idx in enumerate(perm):
        folds[i % k].append(int(doc_idx))
    return folds


def run_cross_conformal(documents, k=5, alpha=0.10, seed=42):
    """Run K-fold cross-conformal prediction.

    For each fold:
      - Use K-1 folds for calibration (compute threshold)
      - Evaluate on held-out fold
    Pool all test-fold coverages.
    """
    folds = create_document_folds(documents, k=k, seed=seed)

    fold_results = []
    all_test_covered = []
    all_test_total = []

    for fold_idx in range(k):
        # Calibration: all folds except fold_idx
        cal_tokens = []
        for i in range(k):
            if i == fold_idx:
                continue
            for doc_id in folds[i]:
                cal_tokens.extend(documents[doc_id]['tokens'])

        # Test: fold_idx only
        test_tokens = []
        for doc_id in folds[fold_idx]:
            test_tokens.extend(documents[doc_id]['tokens'])

        if not cal_tokens or not test_tokens:
            continue

        # Compute calibration threshold
        cal_scores = compute_nonconformity_scores(cal_tokens)
        threshold = compute_threshold(cal_scores, alpha)

        # Evaluate on test fold
        test_scores = compute_nonconformity_scores(test_tokens)
        fold_cov = evaluate_fold_coverage(test_tokens, test_scores, threshold)

        fold_results.append({
            'fold': fold_idx,
            'n_cal_tokens': len(cal_tokens),
            'n_cal_docs': sum(len(folds[i]) for i in range(k) if i != fold_idx),
            'n_test_tokens': len(test_tokens),
            'n_test_docs': len(folds[fold_idx]),
            'threshold': threshold,
            'coverage': fold_cov['coverage'],
        })

        all_test_covered.append(fold_cov['n_covered'])
        all_test_total.append(fold_cov['n_tokens'])

    # Pooled coverage
    total_covered = sum(all_test_covered)
    total_tokens = sum(all_test_total)
    pooled_coverage = total_covered / total_tokens if total_tokens > 0 else 0.0

    # Per-fold coverage stats
    fold_coverages = [fr['coverage'] for fr in fold_results]

    return {
        'k': k,
        'alpha': alpha,
        'fold_results': fold_results,
        'pooled_coverage': pooled_coverage,
        'mean_fold_coverage': float(np.mean(fold_coverages)) if fold_coverages else 0.0,
        'std_fold_coverage': float(np.std(fold_coverages)) if fold_coverages else 0.0,
        'min_fold_coverage': float(np.min(fold_coverages)) if fold_coverages else 0.0,
        'max_fold_coverage': float(np.max(fold_coverages)) if fold_coverages else 0.0,
        'total_tokens': total_tokens,
    }


def get_fixed_split_results(base_dir, dataset_name, seed):
    """Load fixed-split coverage from existing results JSON."""
    seed_dir = os.path.join(base_dir, dataset_name, f'seed_{seed}')
    results_files = [f for f in os.listdir(seed_dir) if f.startswith('results_v2')]
    if not results_files:
        return None
    results_path = os.path.join(seed_dir, results_files[0])
    with open(results_path) as f:
        results = json.load(f)

    fixed = {}
    if 'conformal_softmax' in results:
        lac = results['conformal_softmax'].get('lac', {})
        for cov_key in ['coverage_90', 'coverage_95', 'coverage_99']:
            if cov_key in lac:
                fixed[cov_key] = {
                    'coverage': lac[cov_key]['coverage'],
                    'threshold': lac[cov_key]['threshold'],
                    'avg_set_size': lac[cov_key]['avg_set_size'],
                    'n_samples': lac[cov_key]['n_samples'],
                }
    return fixed


def run_analysis(dataset_name, base_dir, seeds, k=5, alphas=None):
    """Run cross-conformal analysis for a dataset across seeds."""
    if alphas is None:
        alphas = [0.10, 0.05, 0.01]

    all_seed_results = {}

    for seed in seeds:
        json_path = os.path.join(
            base_dir, dataset_name, f'seed_{seed}',
            f'per_token_predictions_{dataset_name}.json')

        if not os.path.exists(json_path):
            print(f"  [SKIP] {json_path} not found")
            continue

        documents = load_per_token_predictions(json_path)
        print(f"\n  Seed {seed}: {len(documents)} documents")

        seed_result = {
            'seed': seed,
            'n_documents': len(documents),
            'cross_conformal': {},
            'fixed_split': get_fixed_split_results(base_dir, dataset_name, seed),
        }

        for alpha in alphas:
            target = 1 - alpha
            cc_result = run_cross_conformal(
                documents, k=k, alpha=alpha, seed=seed)
            cov_key = f"coverage_{int(target * 100)}"
            seed_result['cross_conformal'][cov_key] = cc_result

            print(f"    α={alpha}: Pooled coverage={cc_result['pooled_coverage']*100:.1f}%, "
                  f"Mean±Std={cc_result['mean_fold_coverage']*100:.1f}%"
                  f"±{cc_result['std_fold_coverage']*100:.1f}%")

        all_seed_results[seed] = seed_result

    return all_seed_results


def print_table(results, dataset_name):
    """Print formatted comparison table."""
    seeds = sorted(results.keys())
    coverage_levels = ['coverage_90', 'coverage_95', 'coverage_99']
    target_labels = {'coverage_90': '90%', 'coverage_95': '95%', 'coverage_99': '99%'}

    print(f"\n{'='*80}")
    print(f"CROSS-CONFORMAL vs FIXED-SPLIT: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Main comparison table
    print(f"\n{'Target':>7} {'Seed':>5} | {'Fixed-Split':>12} {'Cross-Conf':>12} "
          f"{'Mean±Std':>16} {'Min':>6} {'Max':>6}")
    print("-" * 75)

    for cov_key in coverage_levels:
        for seed in seeds:
            r = results[seed]
            target = target_labels[cov_key]

            # Fixed-split coverage
            fixed = r.get('fixed_split', {}).get(cov_key, {})
            fixed_cov = fixed.get('coverage', 0.0)

            # Cross-conformal coverage
            cc = r['cross_conformal'].get(cov_key, {})
            pooled = cc.get('pooled_coverage', 0.0)
            mean = cc.get('mean_fold_coverage', 0.0)
            std = cc.get('std_fold_coverage', 0.0)
            min_cov = cc.get('min_fold_coverage', 0.0)
            max_cov = cc.get('max_fold_coverage', 0.0)

            print(f"{target:>7} {seed:>5} | "
                  f"{fixed_cov*100:>11.1f}% "
                  f"{pooled*100:>11.1f}% "
                  f"{mean*100:>7.1f}%±{std*100:>4.1f}% "
                  f"{min_cov*100:>5.1f}% "
                  f"{max_cov*100:>5.1f}%")

    # Summary across seeds
    if len(seeds) > 1:
        print(f"\n{'='*80}")
        print(f"SUMMARY ACROSS SEEDS: {dataset_name.upper()}")
        print(f"{'='*80}")

        for cov_key in coverage_levels:
            target = target_labels[cov_key]
            fixed_vals = []
            cc_vals = []
            for seed in seeds:
                r = results[seed]
                fixed = r.get('fixed_split', {}).get(cov_key, {})
                if fixed:
                    fixed_vals.append(fixed.get('coverage', 0.0))
                cc = r['cross_conformal'].get(cov_key, {})
                if cc:
                    cc_vals.append(cc.get('pooled_coverage', 0.0))

            if fixed_vals and cc_vals:
                print(f"  {target}: Fixed={np.mean(fixed_vals)*100:.1f}%±{np.std(fixed_vals)*100:.1f}%, "
                      f"Cross-Conf={np.mean(cc_vals)*100:.1f}%±{np.std(cc_vals)*100:.1f}%, "
                      f"Δ={np.mean(np.array(cc_vals) - np.array(fixed_vals))*100:+.1f}%")

    # Per-fold detail for seed 42
    if 42 in results:
        print(f"\n{'='*80}")
        print(f"PER-FOLD DETAIL (seed=42, target=90%): {dataset_name.upper()}")
        print(f"{'='*80}")
        cc = results[42]['cross_conformal'].get('coverage_90', {})
        if 'fold_results' in cc:
            print(f"{'Fold':>4} {'Cal Docs':>8} {'Cal Tok':>8} {'Test Docs':>9} "
                  f"{'Test Tok':>8} {'Threshold':>10} {'Coverage':>9}")
            print("-" * 62)
            for fr in cc['fold_results']:
                print(f"{fr['fold']:>4} {fr['n_cal_docs']:>8} "
                      f"{fr['n_cal_tokens']:>8} {fr['n_test_docs']:>9} "
                      f"{fr['n_test_tokens']:>8} {fr['threshold']:>10.6f} "
                      f"{fr['coverage']*100:>8.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Conformal Validation Evaluation")
    parser.add_argument('--base_dir', type=str,
                        default='results/conformal_v2_multiseed',
                        help='Base directory with per-token prediction JSONs')
    parser.add_argument('--datasets', nargs='+', default=['funsd', 'cord'],
                        choices=['funsd', 'cord'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds for cross-conformal')
    parser.add_argument('--output_dir', type=str,
                        default='results/cross_conformal',
                        help='Output directory for JSON results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")

        results = run_analysis(dataset_name, args.base_dir, args.seeds, args.k)
        if results:
            print_table(results, dataset_name)
            all_results[dataset_name] = {
                str(k): v for k, v in results.items()}

    # Save results
    out_path = os.path.join(args.output_dir, 'cross_conformal_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
