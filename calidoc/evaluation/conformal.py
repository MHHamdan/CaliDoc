"""
Conformal Prediction for Document Understanding

This module provides distribution-free uncertainty quantification with
guaranteed coverage rates. Key features:

1. **Coverage Guarantee**: If target coverage is 95%, the prediction sets
   will contain the true label at least 95% of the time (marginal coverage).

2. **Adaptive Set Sizes**: Prediction sets are larger for harder examples,
   providing informative uncertainty estimates.

3. **Selective Prediction**: Reject uncertain predictions to improve
   accuracy on accepted samples.

References:
- Vovk et al., "Algorithmic Learning in a Random World" (2005)
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2022)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ConformalResult:
    """Results from conformal prediction."""
    prediction_sets: List[List[int]]  # List of label sets per sample
    set_sizes: np.ndarray  # Size of each prediction set
    coverage: float  # Empirical coverage
    avg_set_size: float  # Average set size
    threshold: float  # Calibrated threshold used


class ConformalPredictor:
    """
    Conformal predictor for sequence labeling (NER) tasks.

    Provides prediction sets with coverage guarantees using the
    LAC (Least Ambiguous set-valued Classifier) method.
    """

    def __init__(
        self,
        num_classes: int,
        target_coverage: float = 0.95,
        method: str = 'lac',  # 'lac' or 'aps' (Adaptive Prediction Sets)
    ):
        """
        Initialize conformal predictor.

        Args:
            num_classes: Number of output classes
            target_coverage: Target coverage rate (e.g., 0.95 for 95%)
            method: Conformal method ('lac' or 'aps')
        """
        self.num_classes = num_classes
        self.target_coverage = target_coverage
        self.method = method
        self.threshold = None
        self.calibrated = False

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Calibrate the conformal predictor on held-out data.

        Args:
            logits: Model logits [batch, seq_len, num_classes] or [n_samples, num_classes]
            labels: True labels [batch, seq_len] or [n_samples]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            Calibrated threshold
        """
        # Flatten if sequence data
        if logits.dim() == 3:
            batch_size, seq_len, num_classes = logits.shape
            logits = logits.view(-1, num_classes)
            labels = labels.view(-1)
            if mask is not None:
                mask = mask.view(-1)

        # Filter by mask
        if mask is not None:
            valid_idx = mask.bool()
            logits = logits[valid_idx]
            labels = labels[valid_idx]

        # Filter out ignored labels (e.g., -100)
        valid_labels = labels >= 0
        logits = logits[valid_labels]
        labels = labels[valid_labels]

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute nonconformity scores
        scores = self._compute_scores(probs, labels)

        # Compute quantile for target coverage
        n = len(scores)
        quantile_level = np.ceil((n + 1) * self.target_coverage) / n
        quantile_level = min(quantile_level, 1.0)

        self.threshold = np.quantile(scores.cpu().numpy(), quantile_level)
        self.calibrated = True

        print(f"Calibrated threshold: {self.threshold:.4f} for {self.target_coverage*100:.0f}% coverage")

        return self.threshold

    def _compute_scores(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute nonconformity scores.

        For LAC method: score = 1 - prob(true_label)
        For APS method: score = cumsum of sorted probs until true label included
        """
        if self.method == 'lac':
            # LAC: Simple 1 - p(y_true)
            true_probs = probs[torch.arange(len(labels)), labels]
            scores = 1 - true_probs

        elif self.method == 'aps':
            # APS: Cumulative probability until true label is included
            # Reference: Romano et al., "Classification with Valid and Adaptive Coverage"
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Find position of true label in sorted order
            true_positions = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=True)[1]

            # Get cumsum BEFORE the true label position (exclusive)
            # This is sum of probs of all classes ranked higher than true label
            cumsum_before = torch.zeros(len(labels), device=probs.device)
            mask = true_positions > 0
            cumsum_before[mask] = cumsum_probs[torch.arange(len(labels))[mask], true_positions[mask] - 1]

            # Score = cumsum_before + U * prob(true label), where U ~ Uniform(0,1)
            # This gives scores in [cumsum_before, cumsum_before + prob(true label)]
            true_label_probs = sorted_probs[torch.arange(len(labels)), true_positions]
            U = torch.rand_like(true_label_probs)
            scores = cumsum_before + U * true_label_probs

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return scores

    def predict(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> ConformalResult:
        """
        Generate prediction sets with coverage guarantee.

        Args:
            logits: Model logits [batch, seq_len, num_classes] or [n_samples, num_classes]
            mask: Optional attention mask

        Returns:
            ConformalResult with prediction sets and statistics
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        original_shape = logits.shape[:-1]
        is_sequence = logits.dim() == 3

        # Flatten if sequence data
        if is_sequence:
            logits = logits.view(-1, self.num_classes)
            if mask is not None:
                mask = mask.view(-1)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Generate prediction sets
        prediction_sets = []
        set_sizes = []

        for i in range(len(probs)):
            if mask is not None and not mask[i]:
                prediction_sets.append([])
                set_sizes.append(0)
                continue

            prob_vec = probs[i]
            pred_set = self._make_prediction_set(prob_vec)
            prediction_sets.append(pred_set)
            set_sizes.append(len(pred_set))

        set_sizes = np.array(set_sizes)

        # Filter empty sets for statistics
        valid_sizes = set_sizes[set_sizes > 0]

        return ConformalResult(
            prediction_sets=prediction_sets,
            set_sizes=set_sizes,
            coverage=0.0,  # Computed when labels are available
            avg_set_size=valid_sizes.mean() if len(valid_sizes) > 0 else 0,
            threshold=self.threshold,
        )

    def _make_prediction_set(self, probs: torch.Tensor) -> List[int]:
        """
        Construct prediction set for a single sample.
        """
        if self.method == 'lac':
            # Include all classes with prob > 1 - threshold
            threshold_prob = 1 - self.threshold
            pred_set = (probs >= threshold_prob).nonzero(as_tuple=True)[0].tolist()

            # Always include at least the most likely class
            if len(pred_set) == 0:
                pred_set = [probs.argmax().item()]

        elif self.method == 'aps':
            # Include classes in decreasing probability order using randomized criterion
            # Match the calibration scoring: include class k if cumsum_before + U*p_k < threshold
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            pred_set = []
            cumsum_before = 0.0

            for k in range(len(sorted_probs)):
                p_k = sorted_probs[k].item()
                # Randomized inclusion: class k is included if cumsum_before + U*p_k < threshold
                # For deterministic prediction sets, use cumsum_before < threshold
                # (equivalent to U=0, gives smaller sets)
                if cumsum_before < self.threshold:
                    pred_set.append(sorted_indices[k].item())
                    cumsum_before += p_k
                else:
                    break

            # Always include at least the most likely class
            if len(pred_set) == 0:
                pred_set = [sorted_indices[0].item()]

        return pred_set

    def evaluate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate conformal prediction performance.

        Args:
            logits: Model logits
            labels: True labels
            mask: Optional attention mask

        Returns:
            Dictionary with coverage, avg_set_size, etc.
        """
        result = self.predict(logits, mask)

        # Flatten labels if needed
        if labels.dim() > 1:
            labels = labels.view(-1)
        if mask is not None:
            mask = mask.view(-1)

        # Compute coverage (fraction where true label is in prediction set)
        covered = 0
        total = 0

        for i, (pred_set, label) in enumerate(zip(result.prediction_sets, labels.tolist())):
            if mask is not None and not mask[i]:
                continue
            if label < 0:  # Ignored label
                continue

            total += 1
            if label in pred_set:
                covered += 1

        coverage = covered / total if total > 0 else 0

        # Size distribution
        valid_sizes = result.set_sizes[result.set_sizes > 0]

        return {
            'coverage': coverage,
            'target_coverage': self.target_coverage,
            'coverage_gap': coverage - self.target_coverage,
            'avg_set_size': valid_sizes.mean() if len(valid_sizes) > 0 else 0,
            'median_set_size': np.median(valid_sizes) if len(valid_sizes) > 0 else 0,
            'singleton_rate': (valid_sizes == 1).mean() if len(valid_sizes) > 0 else 0,
            'threshold': self.threshold,
            'n_samples': total,
        }


class SelectivePredictor:
    """
    Selective prediction: Abstain on uncertain predictions.

    Achieves higher accuracy by rejecting low-confidence predictions.
    """

    def __init__(self, abstain_threshold: float = 0.5):
        """
        Args:
            abstain_threshold: Confidence threshold below which to abstain
        """
        self.abstain_threshold = abstain_threshold

    def predict(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with abstention.

        Returns:
            predictions: Predicted labels (-1 for abstained)
            confidence: Confidence scores
        """
        probs = F.softmax(logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)

        # Abstain on low-confidence predictions
        abstain_mask = confidence < self.abstain_threshold
        predictions[abstain_mask] = -1

        return predictions, confidence

    def evaluate_at_coverage(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_coverage: float = 0.9,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate accuracy at a target coverage level.

        Args:
            logits: Model logits
            labels: True labels
            target_coverage: Fraction of samples to accept
            mask: Optional attention mask

        Returns:
            Dictionary with accuracy, coverage, threshold
        """
        # Flatten if needed
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            if mask is not None:
                mask = mask.view(-1)

        probs = F.softmax(logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)

        # Filter valid samples
        if mask is not None:
            valid = mask.bool() & (labels >= 0)
        else:
            valid = labels >= 0

        confidence = confidence[valid]
        predictions = predictions[valid]
        labels = labels[valid]

        # Find threshold for target coverage
        n = len(confidence)
        k = int(np.ceil(n * target_coverage))
        sorted_conf, _ = torch.sort(confidence, descending=True)
        threshold = sorted_conf[k - 1].item() if k > 0 else 0

        # Accept samples above threshold
        accepted = confidence >= threshold
        n_accepted = accepted.sum().item()

        # Compute accuracy on accepted samples
        correct = (predictions[accepted] == labels[accepted]).sum().item()
        accuracy = correct / n_accepted if n_accepted > 0 else 0

        return {
            'accuracy': accuracy,
            'coverage': n_accepted / n if n > 0 else 0,
            'threshold': threshold,
            'n_accepted': n_accepted,
            'n_total': n,
        }

    def compute_risk_coverage_curve(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_points: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Compute risk-coverage curve (accuracy vs coverage tradeoff).

        Returns:
            Dictionary with 'coverage', 'accuracy', 'threshold' arrays
        """
        coverages = np.linspace(0.1, 1.0, n_points)
        results = {
            'coverage': [],
            'accuracy': [],
            'threshold': [],
        }

        for target_cov in coverages:
            metrics = self.evaluate_at_coverage(logits, labels, target_cov, mask)
            results['coverage'].append(metrics['coverage'])
            results['accuracy'].append(metrics['accuracy'])
            results['threshold'].append(metrics['threshold'])

        return {k: np.array(v) for k, v in results.items()}


def compute_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and related metrics.

    Args:
        logits: Model logits
        labels: True labels
        n_bins: Number of bins for calibration
        mask: Optional attention mask

    Returns:
        Dictionary with ECE, MCE, reliability diagram data
    """
    # Flatten if needed
    if logits.dim() > 2:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        if mask is not None:
            mask = mask.view(-1)

    probs = F.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)

    # Filter valid samples
    if mask is not None:
        valid = mask.bool() & (labels >= 0)
    else:
        valid = labels >= 0

    confidence = confidence[valid].cpu().numpy()
    predictions = predictions[valid].cpu().numpy()
    labels = labels[valid].cpu().numpy()

    # Compute correctness
    correct = (predictions == labels).astype(float)

    # Bin samples by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    ece = 0
    mce = 0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i + 1])
        bin_count = bin_mask.sum()

        if bin_count > 0:
            bin_accuracy = correct[bin_mask].mean()
            bin_confidence = confidence[bin_mask].mean()

            ece += np.abs(bin_accuracy - bin_confidence) * bin_count
            mce = max(mce, np.abs(bin_accuracy - bin_confidence))

            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)

        bin_counts.append(bin_count)

    ece /= len(confidence)

    return {
        'ece': ece,
        'mce': mce,
        'bin_centers': bin_centers,
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts),
    }
