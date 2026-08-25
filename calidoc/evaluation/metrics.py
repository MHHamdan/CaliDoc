"""
Evaluation Metrics for CaliDoc

Implements metrics for:
- VQA: ANLS (Average Normalized Levenshtein Similarity)
- NER: Entity-level F1, Precision, Recall
- Calibration: ECE, MCE
- Conformal: Coverage, Set Size
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict


def compute_anls(
    prediction: str,
    ground_truths: Union[str, List[str]],
    threshold: float = 0.5
) -> float:
    """
    Compute Average Normalized Levenshtein Similarity (ANLS).

    ANLS is the primary metric for DocVQA. It's the normalized
    Levenshtein similarity clamped at a threshold.

    Args:
        prediction: Model prediction string
        ground_truths: Ground truth answer(s)
        threshold: Threshold below which NLS is set to 0

    Returns:
        ANLS score in [0, 1]
    """
    try:
        from Levenshtein import distance as levenshtein_distance
    except ImportError:
        # Fallback to difflib
        from difflib import SequenceMatcher
        def levenshtein_distance(a, b):
            return int((1 - SequenceMatcher(None, a, b).ratio()) * max(len(a), len(b)))

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    if not prediction or not ground_truths:
        return 0.0

    # Normalize strings
    prediction = prediction.lower().strip()
    ground_truths = [gt.lower().strip() for gt in ground_truths if gt]

    if not ground_truths:
        return 0.0

    max_nls = 0.0
    for gt in ground_truths:
        if not gt:
            continue

        dist = levenshtein_distance(prediction, gt)
        max_len = max(len(prediction), len(gt))

        if max_len == 0:
            nls = 1.0
        else:
            nls = 1.0 - dist / max_len

        # Apply threshold
        if nls < threshold:
            nls = 0.0

        max_nls = max(max_nls, nls)

    return max_nls


def compute_ner_f1(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str],
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute entity-level F1 score for NER.

    Args:
        predictions: Predicted label IDs
        labels: Ground truth label IDs
        id2label: Mapping from IDs to label strings
        ignore_index: Index to ignore

    Returns:
        Dictionary with precision, recall, F1
    """
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score
    except ImportError:
        # Fallback implementation
        return _compute_ner_f1_fallback(predictions, labels, id2label, ignore_index)

    # Convert IDs to label strings
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []

        for p, l in zip(pred_seq, label_seq):
            if l != ignore_index:
                true_seq.append(id2label.get(l, 'O'))
                pred_seq_filtered.append(id2label.get(p, 'O'))

        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)

    return {
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels),
    }


def _compute_ner_f1_fallback(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str],
    ignore_index: int = -100
) -> Dict[str, float]:
    """Fallback F1 computation without seqeval."""
    correct = 0
    pred_count = 0
    true_count = 0

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l == ignore_index:
                continue
            if l != 0:  # Not 'O' tag
                true_count += 1
            if p != 0:
                pred_count += 1
            if p == l and p != 0:
                correct += 1

    precision = correct / max(pred_count, 1)
    recall = correct / max(true_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well-calibrated the model's confidence scores are.
    A well-calibrated model has confidence ≈ accuracy.

    Args:
        probs: Predicted probabilities [n_samples, n_classes]
        labels: Ground truth labels [n_samples]
        n_bins: Number of calibration bins

    Returns:
        ECE value in [0, 1] (lower is better)
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - confidence_in_bin)

    return ece


def compute_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum gap between confidence and accuracy across bins.

    Args:
        probs: Predicted probabilities
        labels: Ground truth labels
        n_bins: Number of bins

    Returns:
        MCE value in [0, 1]
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            max_error = max(max_error, np.abs(accuracy_in_bin - confidence_in_bin))

    return max_error


def compute_coverage(
    prediction_sets: List[List[int]],
    labels: np.ndarray
) -> float:
    """
    Compute empirical coverage of conformal prediction sets.

    Coverage is the fraction of samples where the true label
    is contained in the prediction set.

    Args:
        prediction_sets: List of prediction sets
        labels: Ground truth labels

    Returns:
        Coverage in [0, 1]
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    covered = 0
    for pred_set, label in zip(prediction_sets, labels):
        if label in pred_set:
            covered += 1

    return covered / len(labels)


def compute_set_size(prediction_sets: List[List[int]]) -> Dict[str, float]:
    """
    Compute prediction set size statistics.

    Args:
        prediction_sets: List of prediction sets

    Returns:
        Dictionary with mean, std, min, max set sizes
    """
    sizes = [len(s) for s in prediction_sets]

    return {
        'mean_set_size': np.mean(sizes),
        'std_set_size': np.std(sizes),
        'min_set_size': np.min(sizes),
        'max_set_size': np.max(sizes),
        'empty_set_rate': np.mean([1 if len(s) == 0 else 0 for s in prediction_sets]),
    }


def compute_metrics(
    predictions: Union[List, np.ndarray],
    labels: Union[List, np.ndarray],
    task: str = 'classification',
    id2label: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a task.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task: Task type ('classification', 'vqa', 'ner')
        id2label: Label mapping for NER

    Returns:
        Dictionary with all computed metrics
    """
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(labels, list):
        labels = np.array(labels)

    metrics = {}

    if task == 'classification':
        # Accuracy
        metrics['accuracy'] = np.mean(predictions == labels)

    elif task == 'vqa':
        # For VQA, we expect string predictions and answers
        if isinstance(predictions[0], str):
            anls_scores = [
                compute_anls(pred, label)
                for pred, label in zip(predictions, labels)
            ]
            metrics['anls'] = np.mean(anls_scores)

    elif task == 'ner':
        if id2label is None:
            raise ValueError("id2label required for NER metrics")
        ner_metrics = compute_ner_f1(predictions.tolist(), labels.tolist(), id2label)
        metrics.update(ner_metrics)

    return metrics


def compute_uncertainty_adjusted_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute uncertainty-adjusted accuracy (UAcc).

    UAcc only counts predictions where the model is confident enough.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        confidences: Model confidence scores
        threshold: Confidence threshold

    Returns:
        Dictionary with UAcc and coverage
    """
    confident_mask = confidences >= threshold

    if np.sum(confident_mask) == 0:
        return {'uacc': 0.0, 'confident_ratio': 0.0}

    uacc = np.mean(predictions[confident_mask] == labels[confident_mask])
    confident_ratio = np.mean(confident_mask)

    return {
        'uacc': uacc,
        'confident_ratio': confident_ratio,
        'uacc_times_coverage': uacc * confident_ratio,
    }
