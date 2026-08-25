#!/usr/bin/env python3
"""
Mondrian (Class-Conditional) Conformal Prediction Evaluation

Addresses:
  - Reviewer Q1: per-class coverage analysis
  - Reviewer weakness #1: O-class dominance inflating marginal coverage

Computes:
  1. Per-class LAC thresholds (quantile of nonconformity scores per class)
  2. Per-class coverage under Mondrian (class-conditional) vs marginal thresholds
  3. O-class vs non-O coverage split

Inputs: per-token prediction JSONs from results/conformal_v2_multiseed/
Outputs: JSON results + stdout tables for paper
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calidoc.data.funsd_dataset import FUNSD_LABELS
from calidoc.data.cord_dataset import CORD_LABELS

DATASET_LABELS = {
    'funsd': FUNSD_LABELS,
    'cord': CORD_LABELS,
}


def load_per_token_predictions(json_path: str):
    """Load per-token prediction JSON and flatten to token list."""
    with open(json_path) as f:
        documents = json.load(f)
    tokens = []
    for doc in documents:
        for tok in doc['tokens']:
            tokens.append(tok)
    return tokens, documents


def label_to_id(label_str: str, label_list: list) -> int:
    """Convert label string to integer id."""
    label2id = {l: i for i, l in enumerate(label_list)}
    return label2id.get(label_str, -1)


def compute_nonconformity_scores(tokens):
    """Compute LAC nonconformity scores: 1 - softmax(true_class).

    Since per-token JSONs store `confidence` (max softmax) and `pred_label`,
    we approximate: if pred == true, score = 1 - confidence.
    If pred != true, the true-class probability is unknown but < confidence,
    so we set score = 1.0 (conservative upper bound).
    """
    scores = []
    for tok in tokens:
        if tok['pred_label'] == tok['true_label']:
            scores.append(1.0 - tok['confidence'])
        else:
            # True class probability unavailable; use conservative score
            scores.append(1.0)
    return np.array(scores)


def compute_marginal_threshold(scores, alpha=0.10):
    """Compute marginal LAC threshold at level 1-alpha."""
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method='higher'))


def compute_mondrian_thresholds(tokens, scores, label_list, alpha=0.10):
    """Compute per-class (Mondrian) LAC thresholds."""
    class_scores = defaultdict(list)
    for tok, score in zip(tokens, scores):
        class_scores[tok['true_label']].append(score)

    thresholds = {}
    for label in label_list:
        if label in class_scores and len(class_scores[label]) > 0:
            s = np.array(class_scores[label])
            n = len(s)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_level = min(q_level, 1.0)
            thresholds[label] = float(np.quantile(s, q_level, method='higher'))
        else:
            thresholds[label] = None  # no calibration data for this class
    return thresholds, class_scores


def evaluate_coverage(tokens, scores, threshold, per_class=False):
    """Evaluate coverage using a single marginal threshold."""
    covered = scores <= threshold
    results = {'overall_coverage': float(np.mean(covered)),
               'n_tokens': len(tokens)}

    if per_class:
        class_results = defaultdict(lambda: {'covered': 0, 'total': 0})
        for tok, cov in zip(tokens, covered):
            label = tok['true_label']
            class_results[label]['total'] += 1
            if cov:
                class_results[label]['covered'] += 1
        for label in class_results:
            d = class_results[label]
            d['coverage'] = d['covered'] / d['total'] if d['total'] > 0 else 0.0
        results['per_class'] = dict(class_results)

    return results


def evaluate_mondrian_coverage(tokens, scores, class_thresholds):
    """Evaluate coverage using per-class Mondrian thresholds."""
    class_results = defaultdict(lambda: {'covered': 0, 'total': 0})
    total_covered = 0

    for tok, score in zip(tokens, scores):
        label = tok['true_label']
        class_results[label]['total'] += 1
        threshold = class_thresholds.get(label)
        if threshold is not None and score <= threshold:
            class_results[label]['covered'] += 1
            total_covered += 1

    for label in class_results:
        d = class_results[label]
        d['coverage'] = d['covered'] / d['total'] if d['total'] > 0 else 0.0

    overall = total_covered / len(tokens) if len(tokens) > 0 else 0.0
    return {'overall_coverage': overall, 'per_class': dict(class_results),
            'n_tokens': len(tokens)}


def compute_set_sizes_from_threshold(tokens, threshold):
    """Approximate prediction set sizes given a LAC threshold.

    For LAC, a class c is in the prediction set if its softmax prob >= 1 - threshold.
    We only know the top-1 confidence, so we estimate:
      - If confidence >= 1 - threshold: at least 1 class (the predicted one) is in the set
      - Set size ≈ 1 for high-confidence tokens under typical thresholds
    For a rough estimate, use the APS set sizes from the JSON as ground truth.
    """
    set_sizes = []
    for tok in tokens:
        # Use APS set sizes as proxy since LAC sets are often stored empty
        set_sizes.append(tok.get('set_size_aps', 1))
    return np.array(set_sizes)


def run_analysis(dataset_name, base_dir, seeds, alpha=0.10):
    """Run full Mondrian CP analysis for a dataset across seeds."""
    label_list = DATASET_LABELS[dataset_name]
    all_seed_results = {}

    for seed in seeds:
        json_path = os.path.join(
            base_dir, dataset_name, f'seed_{seed}',
            f'per_token_predictions_{dataset_name}.json')

        if not os.path.exists(json_path):
            print(f"  [SKIP] {json_path} not found")
            continue

        tokens, documents = load_per_token_predictions(json_path)
        print(f"\n  Seed {seed}: {len(tokens)} tokens, {len(documents)} documents")

        # Split into calibration and test sets
        # Use 70/30 document-level split (same as original pipeline for FUNSD)
        np.random.seed(seed)
        n_docs = len(documents)
        perm = np.random.permutation(n_docs)
        n_cal = int(n_docs * 0.7)
        cal_doc_ids = set(perm[:n_cal].tolist())

        cal_tokens = []
        test_tokens = []
        for doc in documents:
            for tok in doc['tokens']:
                if doc['doc_id'] in cal_doc_ids:
                    cal_tokens.append(tok)
                else:
                    test_tokens.append(tok)

        print(f"    Calibration: {len(cal_tokens)} tokens ({n_cal} docs)")
        print(f"    Test: {len(test_tokens)} tokens ({n_docs - n_cal} docs)")

        # Compute scores
        cal_scores = compute_nonconformity_scores(cal_tokens)
        test_scores = compute_nonconformity_scores(test_tokens)

        # Marginal threshold
        marginal_threshold = compute_marginal_threshold(cal_scores, alpha)
        print(f"    Marginal threshold (α={alpha}): {marginal_threshold:.6f}")

        # Mondrian thresholds
        mondrian_thresholds, cal_class_scores = compute_mondrian_thresholds(
            cal_tokens, cal_scores, label_list, alpha)

        # Evaluate marginal coverage
        marginal_results = evaluate_coverage(
            test_tokens, test_scores, marginal_threshold, per_class=True)

        # Evaluate Mondrian coverage
        mondrian_results = evaluate_mondrian_coverage(
            test_tokens, test_scores, mondrian_thresholds)

        # O-class vs non-O split
        o_tokens_marginal = []
        non_o_tokens_marginal = []
        o_tokens_mondrian = []
        non_o_tokens_mondrian = []

        for tok, score in zip(test_tokens, test_scores):
            is_o = tok['true_label'] == 'O'
            m_covered = score <= marginal_threshold
            mon_threshold = mondrian_thresholds.get(tok['true_label'])
            mon_covered = mon_threshold is not None and score <= mon_threshold

            if is_o:
                o_tokens_marginal.append(m_covered)
                o_tokens_mondrian.append(mon_covered)
            else:
                non_o_tokens_marginal.append(m_covered)
                non_o_tokens_mondrian.append(mon_covered)

        seed_result = {
            'seed': seed,
            'n_cal': len(cal_tokens),
            'n_test': len(test_tokens),
            'alpha': alpha,
            'marginal_threshold': marginal_threshold,
            'mondrian_thresholds': {
                k: v for k, v in mondrian_thresholds.items() if v is not None},
            'marginal_coverage': marginal_results['overall_coverage'],
            'mondrian_coverage': mondrian_results['overall_coverage'],
            'marginal_per_class': {},
            'mondrian_per_class': {},
            'o_vs_non_o': {
                'o_marginal_coverage': float(np.mean(o_tokens_marginal)) if o_tokens_marginal else 0.0,
                'o_mondrian_coverage': float(np.mean(o_tokens_mondrian)) if o_tokens_mondrian else 0.0,
                'non_o_marginal_coverage': float(np.mean(non_o_tokens_marginal)) if non_o_tokens_marginal else 0.0,
                'non_o_mondrian_coverage': float(np.mean(non_o_tokens_mondrian)) if non_o_tokens_mondrian else 0.0,
                'n_o': len(o_tokens_marginal),
                'n_non_o': len(non_o_tokens_marginal),
            },
        }

        # Build per-class results
        for label in label_list:
            m_class = marginal_results['per_class'].get(label, {'coverage': 0.0, 'total': 0})
            mon_class = mondrian_results['per_class'].get(label, {'coverage': 0.0, 'total': 0})
            n_cal_class = len(cal_class_scores.get(label, []))
            seed_result['marginal_per_class'][label] = {
                'coverage': m_class.get('coverage', 0.0),
                'n_tokens': m_class.get('total', 0),
            }
            seed_result['mondrian_per_class'][label] = {
                'coverage': mon_class.get('coverage', 0.0),
                'n_tokens': mon_class.get('total', 0),
                'n_cal': n_cal_class,
                'threshold': mondrian_thresholds.get(label),
            }

        all_seed_results[seed] = seed_result

    return all_seed_results


def print_table(results, dataset_name, alpha):
    """Print formatted results table."""
    label_list = DATASET_LABELS[dataset_name]
    seeds = sorted(results.keys())

    target = 1 - alpha
    print(f"\n{'='*80}")
    print(f"MONDRIAN CP RESULTS: {dataset_name.upper()} (target coverage: {target*100:.0f}%)")
    print(f"{'='*80}")

    # Per-class table header
    header = f"{'Class':<25} {'n_tokens':>8}"
    for seed in seeds:
        header += f" | Marg_{seed:>3} Mond_{seed:>3}"
    print(header)
    print("-" * len(header))

    for label in label_list:
        row = f"{label:<25}"
        # Average n_tokens across seeds
        n_tok = int(np.mean([results[s]['marginal_per_class'].get(label, {}).get('n_tokens', 0)
                             for s in seeds]))
        row += f" {n_tok:>8}"
        for seed in seeds:
            m_cov = results[seed]['marginal_per_class'].get(label, {}).get('coverage', 0.0)
            mon_cov = results[seed]['mondrian_per_class'].get(label, {}).get('coverage', 0.0)
            row += f" | {m_cov*100:>5.1f}% {mon_cov*100:>5.1f}%"
        print(row)

    # O vs non-O summary
    print(f"\n{'='*80}")
    print("O-CLASS vs NON-O SUMMARY")
    print(f"{'='*80}")
    header2 = f"{'Split':<15} {'n_tokens':>8}"
    for seed in seeds:
        header2 += f" | Marg_{seed:>3} Mond_{seed:>3}"
    print(header2)
    print("-" * len(header2))

    for split_name, key_m, key_mon, key_n in [
        ("O-class", "o_marginal_coverage", "o_mondrian_coverage", "n_o"),
        ("Non-O", "non_o_marginal_coverage", "non_o_mondrian_coverage", "n_non_o"),
    ]:
        row = f"{split_name:<15}"
        n = int(np.mean([results[s]['o_vs_non_o'][key_n] for s in seeds]))
        row += f" {n:>8}"
        for seed in seeds:
            m = results[seed]['o_vs_non_o'][key_m]
            mon = results[seed]['o_vs_non_o'][key_mon]
            row += f" | {m*100:>5.1f}% {mon*100:>5.1f}%"
        print(row)

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL COVERAGE")
    print(f"{'='*80}")
    for seed in seeds:
        r = results[seed]
        print(f"  Seed {seed}: Marginal={r['marginal_coverage']*100:.1f}%, "
              f"Mondrian={r['mondrian_coverage']*100:.1f}%")

    # Mean ± std across seeds
    if len(seeds) > 1:
        m_covs = [results[s]['marginal_coverage'] for s in seeds]
        mon_covs = [results[s]['mondrian_coverage'] for s in seeds]
        print(f"\n  Mean Marginal: {np.mean(m_covs)*100:.1f}% ± {np.std(m_covs)*100:.1f}%")
        print(f"  Mean Mondrian: {np.mean(mon_covs)*100:.1f}% ± {np.std(mon_covs)*100:.1f}%")

        # Mean O vs non-O
        for split_name, key_m, key_mon in [
            ("O-class", "o_marginal_coverage", "o_mondrian_coverage"),
            ("Non-O", "non_o_marginal_coverage", "non_o_mondrian_coverage"),
        ]:
            m_vals = [results[s]['o_vs_non_o'][key_m] for s in seeds]
            mon_vals = [results[s]['o_vs_non_o'][key_mon] for s in seeds]
            print(f"  {split_name}: Marginal={np.mean(m_vals)*100:.1f}%±{np.std(m_vals)*100:.1f}%, "
                  f"Mondrian={np.mean(mon_vals)*100:.1f}%±{np.std(mon_vals)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Mondrian (Class-Conditional) Conformal Prediction Evaluation")
    parser.add_argument('--base_dir', type=str,
                        default='results/conformal_v2_multiseed',
                        help='Base directory with per-token prediction JSONs')
    parser.add_argument('--datasets', nargs='+', default=['funsd', 'cord'],
                        choices=['funsd', 'cord'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--alpha', type=float, default=0.10,
                        help='Significance level (default: 0.10 for 90%% coverage)')
    parser.add_argument('--output_dir', type=str,
                        default='results/mondrian_cp',
                        help='Output directory for JSON results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")

        results = run_analysis(dataset_name, args.base_dir, args.seeds, args.alpha)
        if results:
            print_table(results, dataset_name, args.alpha)
            all_results[dataset_name] = {
                str(k): v for k, v in results.items()}

    # Save results
    out_path = os.path.join(args.output_dir, 'mondrian_cp_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
