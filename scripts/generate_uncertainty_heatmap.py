#!/usr/bin/env python3
"""
Generate Uncertainty Heatmap Visualizations for CaliDoc Paper

Two modes:
  1. --from_per_token: Uses per-token prediction JSON from evaluate_conformal_v2.py
     to build Error-vs-Uncertainty heatmap (entity class x confidence bucket).

  2. --from_existing: Uses existing calibration JSON results to build a
     calibration-gap heatmap (dataset x confidence bins).

Usage:
    # Full heatmap from per-token predictions (after nibi run):
    python scripts/generate_uncertainty_heatmap.py \
        --from_per_token \
        --funsd_json results/conformal_v2_funsd/per_token_predictions_funsd.json \
        --cord_json results/conformal_v2_cord/per_token_predictions_cord.json \
        --output docs/paper/figures/uncertainty_heatmap.pdf

    # Fallback from existing calibration data:
    python scripts/generate_uncertainty_heatmap.py \
        --from_existing \
        --funsd_results results/conformal_funsd_6915303/results_20260114_202425.json \
        --cord_results results/conformal_cord_6682735/results_20260111_213829.json \
        --output docs/paper/figures/uncertainty_heatmap.pdf
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option 1: Error-vs-Uncertainty heatmap from per-token predictions
# ---------------------------------------------------------------------------

CONFIDENCE_BUCKETS = [
    (0.0, 0.5, '0-50%'),
    (0.5, 0.8, '50-80%'),
    (0.8, 0.95, '80-95%'),
    (0.95, 0.99, '95-99%'),
    (0.99, 1.001, '99-100%'),
]


def load_per_token_json(path):
    """Load per-token predictions JSON."""
    with open(path) as f:
        return json.load(f)


def build_error_uncertainty_matrix(docs, entity_classes=None):
    """Build error rate matrix: rows=entity classes, cols=confidence buckets."""
    # Collect (true_label, confidence, is_error) tuples
    records = []
    for doc in docs:
        for tok in doc.get('tokens', []):
            true_label = tok['true_label']
            conf = tok['confidence']
            is_error = 1 if tok['pred_label'] != tok['true_label'] else 0
            records.append((true_label, conf, is_error))

    if not records:
        return None, None, None

    # Determine entity classes
    all_labels = sorted(set(r[0] for r in records))
    # Filter to non-O labels, take top N by count
    label_counts = defaultdict(int)
    for r in records:
        label_counts[r[0]] += 1

    if entity_classes is None:
        # Use non-O B- labels sorted by frequency
        entity_classes = [l for l in all_labels
                          if l != 'O' and l.startswith('B-')]
        entity_classes = sorted(entity_classes,
                                key=lambda x: label_counts.get(x, 0),
                                reverse=True)[:5]
        # Also include O
        entity_classes = ['O'] + entity_classes

    n_rows = len(entity_classes)
    n_cols = len(CONFIDENCE_BUCKETS)
    error_matrix = np.full((n_rows, n_cols), np.nan)
    count_matrix = np.zeros((n_rows, n_cols), dtype=int)

    for row_idx, cls_name in enumerate(entity_classes):
        for col_idx, (lo, hi, _) in enumerate(CONFIDENCE_BUCKETS):
            matching = [r for r in records
                        if r[0] == cls_name and lo <= r[1] < hi]
            if matching:
                errors = sum(r[2] for r in matching)
                error_matrix[row_idx, col_idx] = errors / len(matching)
                count_matrix[row_idx, col_idx] = len(matching)

    return error_matrix, count_matrix, entity_classes


def plot_error_uncertainty_heatmap(matrices, output_path):
    """Plot side-by-side heatmaps for FUNSD and CORD."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping plot")
        return

    n_datasets = len(matrices)
    # Vertical stacking for better readability
    fig, axes = plt.subplots(n_datasets, 1,
                              figsize=(10, 4.5 * n_datasets))
    if n_datasets == 1:
        axes = [axes]

    bucket_labels = [b[2] for b in CONFIDENCE_BUCKETS]

    for ax, (dataset_name, error_matrix, count_matrix, classes) in \
            zip(axes, matrices):
        # Mask NaN cells
        masked = np.ma.masked_invalid(error_matrix)
        cmap = plt.cm.RdYlGn_r  # red=high error, green=low error
        cmap.set_bad(color='white')

        im = ax.imshow(masked, cmap=cmap, aspect='auto',
                       vmin=0, vmax=1)

        # Add grid lines for cell borders
        for i in range(error_matrix.shape[0] + 1):
            ax.axhline(i - 0.5, color='white', linewidth=2)
        for j in range(error_matrix.shape[1] + 1):
            ax.axvline(j - 0.5, color='white', linewidth=2)

        # Annotate cells
        for i in range(error_matrix.shape[0]):
            for j in range(error_matrix.shape[1]):
                if not np.isnan(error_matrix[i, j]):
                    err_pct = error_matrix[i, j] * 100
                    n = count_matrix[i, j]
                    text_color = 'white' if error_matrix[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{err_pct:.0f}%\n(n={n})',
                            ha='center', va='center', fontsize=9,
                            color=text_color, fontweight='bold')

        # Clean class names for display
        display_classes = [c.replace('B-', '').replace('I-', '')
                           for c in classes]

        ax.set_xticks(range(len(bucket_labels)))
        ax.set_xticklabels(bucket_labels, fontsize=10)
        ax.set_yticks(range(len(display_classes)))
        ax.set_yticklabels(display_classes, fontsize=10)
        ax.set_xlabel('Confidence Bucket', fontsize=11)
        ax.set_ylabel('Entity Class', fontsize=11)
        ax.set_title(f'{dataset_name} — Error Rate by Class & Confidence',
                     fontsize=12, fontweight='bold')

    # Place colorbar at bottom to avoid overlap with heatmap cells
    cbar_ax = fig.add_axes([0.25, -0.03, 0.5, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Error Rate', orientation='horizontal')
    fig.subplots_adjust(hspace=0.4)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved heatmap to {output_path}")


# ---------------------------------------------------------------------------
# Option 2: Calibration-gap heatmap from existing results
# ---------------------------------------------------------------------------

def build_calibration_gap_heatmap(results_paths, output_path):
    """Build calibration-gap heatmap from existing result JSONs."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping plot")
        return

    datasets_data = []
    for dataset_name, path in results_paths.items():
        if not os.path.exists(path):
            logger.warning(f"Results file not found: {path}")
            continue
        with open(path) as f:
            results = json.load(f)

        cal = results.get('calibration', {})
        bin_accs = np.array(cal.get('bin_accuracies', []))
        bin_confs = np.array(cal.get('bin_confidences', []))
        bin_counts = np.array(cal.get('bin_counts', []))
        bin_centers = np.array(cal.get('bin_centers', []))

        if len(bin_accs) == 0:
            continue

        # Calibration gap = confidence - accuracy (positive = overconfident)
        gaps = bin_confs - bin_accs
        datasets_data.append((dataset_name, bin_centers, gaps, bin_counts,
                              bin_accs, bin_confs))

    if not datasets_data:
        logger.warning("No valid calibration data found")
        return

    fig, axes = plt.subplots(1, len(datasets_data),
                              figsize=(6 * len(datasets_data), 4))
    if len(datasets_data) == 1:
        axes = [axes]

    for ax, (name, centers, gaps, counts, accs, confs) in \
            zip(axes, datasets_data):
        n_bins = len(centers)
        colors = ['#E74C3C' if g > 0.05 else '#F39C12' if g > 0.01
                   else '#2ECC71' for g in gaps]

        bars = ax.bar(range(n_bins), gaps, color=colors, alpha=0.8,
                      edgecolor='#333', linewidth=0.5)

        # Annotate with count
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'n={int(count)}', ha='center', va='bottom',
                        fontsize=6, rotation=45)

        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([f'{c:.2f}' for c in centers], fontsize=7,
                           rotation=45)
        ax.set_xlabel('Confidence Bin Center', fontsize=9)
        ax.set_ylabel('Calibration Gap (conf - acc)', fontsize=9)
        ax.set_title(f'{name} Calibration Gap', fontsize=10,
                     fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(-0.3, 0.5)

        ece = results.get('calibration', {}).get('ece', 0)
        ax.text(0.02, 0.98, f'ECE: {ece*100:.2f}%',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved calibration-gap heatmap to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate uncertainty heatmap for CaliDoc paper")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--from_per_token', action='store_true',
                      help='Build from per-token prediction JSONs')
    mode.add_argument('--from_existing', action='store_true',
                      help='Build from existing calibration result JSONs')

    parser.add_argument('--funsd_json', type=str, default=None,
                        help='Per-token JSON for FUNSD')
    parser.add_argument('--cord_json', type=str, default=None,
                        help='Per-token JSON for CORD')

    parser.add_argument('--funsd_results', type=str, default=None,
                        help='Existing results JSON for FUNSD')
    parser.add_argument('--cord_results', type=str, default=None,
                        help='Existing results JSON for CORD')

    parser.add_argument('--output', type=str,
                        default='docs/paper/figures/uncertainty_heatmap.pdf')
    args = parser.parse_args()

    if args.from_per_token:
        matrices = []
        for dataset_name, json_path in [('FUNSD', args.funsd_json),
                                         ('CORD', args.cord_json)]:
            if json_path is None or not os.path.exists(json_path):
                logger.info(f"Skipping {dataset_name} (no JSON)")
                continue
            docs = load_per_token_json(json_path)
            error_matrix, count_matrix, classes = \
                build_error_uncertainty_matrix(docs)
            if error_matrix is not None:
                matrices.append((dataset_name, error_matrix,
                                 count_matrix, classes))

        if matrices:
            plot_error_uncertainty_heatmap(matrices, args.output)
        else:
            logger.warning("No data to plot")

    elif args.from_existing:
        results_paths = {}
        if args.funsd_results:
            results_paths['FUNSD'] = args.funsd_results
        if args.cord_results:
            results_paths['CORD'] = args.cord_results

        if results_paths:
            build_calibration_gap_heatmap(results_paths, args.output)
        else:
            logger.warning("No result paths provided")


if __name__ == '__main__':
    main()
