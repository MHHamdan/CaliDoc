"""
Conformal Prediction Metrics

Evaluation utilities for conformal prediction including:
- Coverage analysis (marginal and conditional)
- Set size efficiency
- Reliability diagrams
- Calibration visualization
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


def evaluate_conformal_predictions(
    prediction_sets: List[List[int]],
    labels: np.ndarray,
    confidences: np.ndarray,
    alpha: float = 0.1
) -> Dict[str, float]:
    """
    Comprehensive evaluation of conformal predictions.

    Args:
        prediction_sets: List of prediction sets
        labels: Ground truth labels
        confidences: Model confidence scores
        alpha: Target miscoverage rate

    Returns:
        Dictionary with all conformal metrics
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.numpy()

    metrics = {}

    # Coverage
    covered = [1 if label in pred_set else 0 for pred_set, label in zip(prediction_sets, labels)]
    metrics['coverage'] = np.mean(covered)
    metrics['coverage_gap'] = metrics['coverage'] - (1 - alpha)

    # Set size statistics
    sizes = [len(s) for s in prediction_sets]
    metrics['avg_set_size'] = np.mean(sizes)
    metrics['median_set_size'] = np.median(sizes)
    metrics['std_set_size'] = np.std(sizes)

    # Empty sets
    metrics['empty_set_rate'] = np.mean([1 if len(s) == 0 else 0 for s in prediction_sets])

    # Single element sets (ideal)
    metrics['single_set_rate'] = np.mean([1 if len(s) == 1 else 0 for s in prediction_sets])

    # Conditional coverage by confidence bins
    cond_coverage = compute_conditional_coverage(
        prediction_sets, labels, confidences, n_bins=5
    )
    metrics['conditional_coverage_std'] = np.std(list(cond_coverage.values()))

    # Efficiency: penalize large sets
    metrics['efficiency'] = metrics['coverage'] / (1 + np.log(metrics['avg_set_size'] + 1))

    return metrics


def compute_conditional_coverage(
    prediction_sets: List[List[int]],
    labels: np.ndarray,
    conditioning_variable: np.ndarray,
    n_bins: int = 5
) -> Dict[str, float]:
    """
    Compute coverage conditioned on a variable (e.g., confidence).

    Args:
        prediction_sets: List of prediction sets
        labels: Ground truth labels
        conditioning_variable: Variable to condition on
        n_bins: Number of bins

    Returns:
        Dictionary mapping bin labels to coverage values
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(conditioning_variable, torch.Tensor):
        conditioning_variable = conditioning_variable.numpy()

    bin_edges = np.percentile(conditioning_variable, np.linspace(0, 100, n_bins + 1))
    conditional_coverage = {}

    for i in range(n_bins):
        mask = (conditioning_variable >= bin_edges[i]) & (conditioning_variable < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper edge in last bin
            mask = (conditioning_variable >= bin_edges[i]) & (conditioning_variable <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            covered = [
                1 if labels[j] in prediction_sets[j] else 0
                for j in np.where(mask)[0]
            ]
            bin_label = f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}'
            conditional_coverage[bin_label] = np.mean(covered)

    return conditional_coverage


def compute_set_size_by_difficulty(
    prediction_sets: List[List[int]],
    difficulty_scores: np.ndarray,
    n_bins: int = 5
) -> Dict[str, float]:
    """
    Analyze set size as a function of sample difficulty.

    Args:
        prediction_sets: List of prediction sets
        difficulty_scores: Difficulty estimates (e.g., OCR difficulty)
        n_bins: Number of bins

    Returns:
        Dictionary mapping difficulty bins to average set sizes
    """
    if isinstance(difficulty_scores, torch.Tensor):
        difficulty_scores = difficulty_scores.numpy()

    bin_edges = np.percentile(difficulty_scores, np.linspace(0, 100, n_bins + 1))
    set_sizes_by_difficulty = {}

    for i in range(n_bins):
        mask = (difficulty_scores >= bin_edges[i]) & (difficulty_scores < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (difficulty_scores >= bin_edges[i]) & (difficulty_scores <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            sizes = [len(prediction_sets[j]) for j in np.where(mask)[0]]
            bin_label = f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}'
            set_sizes_by_difficulty[bin_label] = np.mean(sizes)

    return set_sizes_by_difficulty


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = 'Reliability Diagram',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot reliability diagram showing calibration.

    Args:
        probs: Predicted probabilities [n_samples, n_classes]
        labels: Ground truth labels [n_samples]
        n_bins: Number of calibration bins
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.7,
            label='Accuracy', edgecolor='black')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Confidence histogram
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.7,
            color='orange', edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_coverage_vs_alpha(
    all_scores: np.ndarray,
    labels: np.ndarray,
    alpha_values: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot coverage vs alpha (miscoverage rate).

    Args:
        all_scores: Nonconformity scores [n_samples]
        labels: Ground truth labels
        alpha_values: List of alpha values to evaluate
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    coverages = []
    set_sizes = []

    for alpha in alpha_values:
        # Compute threshold
        n = len(all_scores)
        quantile_idx = int(np.ceil((n + 1) * (1 - alpha)))
        threshold = np.sort(all_scores)[min(quantile_idx, n-1)]

        # Compute coverage
        covered = np.sum(all_scores <= threshold) / n
        coverages.append(covered)

        # Compute set size (simplified)
        set_sizes.append(1 + alpha * 5)  # Placeholder

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(alpha_values, coverages, 'b-o', label='Empirical Coverage')
    ax1.plot(alpha_values, [1 - a for a in alpha_values], 'r--', label='Target (1-α)')
    ax1.set_xlabel('α (Miscoverage Rate)')
    ax1.set_ylabel('Coverage')
    ax1.legend()
    ax1.set_title('Coverage vs α')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_set_size_histogram(
    prediction_sets: List[List[int]],
    title: str = 'Prediction Set Size Distribution',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of prediction set sizes.

    Args:
        prediction_sets: List of prediction sets
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    sizes = [len(s) for s in prediction_sets]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(sizes, bins=range(0, max(sizes) + 2), alpha=0.7,
            edgecolor='black', align='left')
    ax.set_xlabel('Set Size')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.axvline(np.mean(sizes), color='r', linestyle='--',
               label=f'Mean: {np.mean(sizes):.2f}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
