#!/usr/bin/env python3
"""
Aggregate multi-seed experiment results and compute confidence intervals.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import argparse

def load_results(output_dir: Path, dataset: str, seeds: list) -> dict:
    """Load results from multiple seeds."""
    results = {
        'f1': [],
        'precision': [],
        'recall': [],
        'anls': [],  # for DocVQA
    }

    for seed in seeds:
        result_dir = output_dir / f"multiseed_{dataset}_seed{seed}"

        # Try to find result files
        for result_file in result_dir.glob("**/results*.json"):
            with open(result_file) as f:
                data = json.load(f)

            if 'f1' in data:
                results['f1'].append(data['f1'])
            if 'precision' in data:
                results['precision'].append(data['precision'])
            if 'recall' in data:
                results['recall'].append(data['recall'])
            if 'anls' in data:
                results['anls'].append(data['anls'])
            break

    return results

def compute_confidence_interval(values: list, confidence: float = 0.95) -> tuple:
    """Compute mean and confidence interval."""
    if len(values) < 2:
        return np.mean(values), 0.0, 0.0

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)

    # t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * std / np.sqrt(n)

    return mean, std, margin

def format_result(mean: float, margin: float, is_percentage: bool = True) -> str:
    """Format result as mean +/- margin."""
    if is_percentage:
        return f"{mean*100:.2f} $\\pm$ {margin*100:.2f}"
    return f"{mean:.4f} $\\pm$ {margin:.4f}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    seeds = args.seeds

    print("=" * 60)
    print("Multi-Seed Results Summary")
    print("=" * 60)
    print(f"Seeds: {seeds}")
    print()

    datasets = ['funsd', 'cord', 'docvqa']

    for dataset in datasets:
        print(f"\n{dataset.upper()}")
        print("-" * 40)

        results = load_results(output_dir, dataset, seeds)

        if results['f1']:
            mean, std, margin = compute_confidence_interval(results['f1'])
            print(f"  F1: {format_result(mean, margin)}")
            print(f"      Raw values: {[f'{v*100:.2f}' for v in results['f1']]}")

        if results['precision']:
            mean, std, margin = compute_confidence_interval(results['precision'])
            print(f"  Precision: {format_result(mean, margin)}")

        if results['recall']:
            mean, std, margin = compute_confidence_interval(results['recall'])
            print(f"  Recall: {format_result(mean, margin)}")

        if results['anls']:
            mean, std, margin = compute_confidence_interval(results['anls'])
            print(f"  ANLS: {format_result(mean, margin)}")

    print("\n" + "=" * 60)
    print("LaTeX Table Format:")
    print("=" * 60)

    print("""
\\begin{table}[t]
\\centering
\\caption{Results with 95\\% confidence intervals (3 seeds)}
\\begin{tabular}{lcc}
\\toprule
Dataset & Metric & Result \\\\
\\midrule""")

    for dataset in datasets:
        results = load_results(output_dir, dataset, seeds)
        metric = 'f1' if dataset != 'docvqa' else 'anls'
        metric_name = 'F1' if metric == 'f1' else 'ANLS'

        if results[metric]:
            mean, std, margin = compute_confidence_interval(results[metric])
            print(f"{dataset.upper()} & {metric_name} & {format_result(mean, margin)}\\% \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

if __name__ == '__main__':
    main()
