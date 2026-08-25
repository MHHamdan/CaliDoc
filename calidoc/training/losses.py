"""
Loss Functions for CaliDoc Training

Implements multi-task losses for:
- Task-specific loss (VQA, NER)
- Calibration loss (ECE minimization)
- Quality estimation loss
- Routing loss with cost optimization
- Focal loss for class imbalance
- Label smoothing for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in NER tasks.

    Focal loss down-weights easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weight (default: None for uniform)
        ignore_index: Index to ignore
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        # Flatten for NER
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

        # Filter ignored indices
        mask = labels != self.ignore_index
        logits = logits[mask]
        labels = labels[mask]

        if len(labels) == 0:
            return torch.tensor(0.0, device=logits.device)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        # Get probability of true class
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting if specified
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha

        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


def compute_task_loss_with_focal(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str = 'ner',
    ignore_index: int = -100,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute task-specific loss with optional focal loss and label smoothing.

    Args:
        logits: Model predictions
        labels: Ground truth labels
        task_type: Type of task ('classification', 'vqa', 'ner')
        ignore_index: Index to ignore
        use_focal: Whether to use focal loss
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Task loss tensor
    """
    if task_type == 'ner' and use_focal:
        focal_loss = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)
        return focal_loss(logits, labels)

    if task_type == 'ner' and label_smoothing > 0:
        # Flatten for NER
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        return F.cross_entropy(
            logits_flat, labels_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

    # Default: standard cross-entropy
    if task_type == 'ner':
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index
        )

    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def compute_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str = 'classification',
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute task-specific loss.

    Args:
        logits: Model predictions
        labels: Ground truth labels
        task_type: Type of task ('classification', 'vqa', 'ner')
        ignore_index: Index to ignore in loss computation

    Returns:
        Task loss tensor
    """
    if task_type == 'vqa':
        # VQA with start/end positions
        if logits.dim() == 3 and logits.shape[-1] == 2:
            start_logits = logits[:, :, 0]
            end_logits = logits[:, :, 1]

            start_loss = F.cross_entropy(
                start_logits,
                labels[:, 0],
                ignore_index=ignore_index
            )
            end_loss = F.cross_entropy(
                end_logits,
                labels[:, 1],
                ignore_index=ignore_index
            )
            return (start_loss + end_loss) / 2

    elif task_type == 'ner':
        # Token classification
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index
        )

    else:
        # Standard classification
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=ignore_index
        )


def compute_calibration_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute Expected Calibration Error (ECE) as a differentiable loss.

    This implements a soft-binning version of ECE that can be
    backpropagated through.

    Args:
        logits: Model logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        n_bins: Number of calibration bins
        temperature: Temperature for softmax

    Returns:
        ECE loss tensor
    """
    # Apply temperature scaling
    probs = F.softmax(logits / temperature, dim=-1)
    confidences, predictions = probs.max(dim=-1)

    # Compute accuracy
    accuracies = (predictions == labels).float()

    # Soft binning
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        # Soft assignment to bin
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Create soft bin membership using sigmoid
        in_bin_soft = torch.sigmoid(10 * (confidences - lower)) * \
                      torch.sigmoid(10 * (upper - confidences))

        bin_weight = in_bin_soft.sum() / (len(confidences) + 1e-10)

        if bin_weight > 0:
            bin_accuracy = (in_bin_soft * accuracies).sum() / (in_bin_soft.sum() + 1e-10)
            bin_confidence = (in_bin_soft * confidences).sum() / (in_bin_soft.sum() + 1e-10)
            ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)

    return ece.squeeze()


def compute_quality_loss(
    predicted_quality: torch.Tensor,
    target_quality: torch.Tensor,
    predicted_ocr_difficulty: Optional[torch.Tensor] = None,
    target_ocr_difficulty: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute quality estimation loss.

    Args:
        predicted_quality: Predicted quality scores
        target_quality: Target quality scores (from OCR accuracy)
        predicted_ocr_difficulty: Predicted OCR difficulty
        target_ocr_difficulty: Target OCR difficulty

    Returns:
        Quality loss tensor
    """
    loss = F.mse_loss(predicted_quality, target_quality)

    if predicted_ocr_difficulty is not None and target_ocr_difficulty is not None:
        loss += F.mse_loss(predicted_ocr_difficulty, target_ocr_difficulty)

    return loss


def compute_routing_loss(
    path_weights: torch.Tensor,
    task_losses: torch.Tensor,
    path_costs: torch.Tensor,
    cost_weight: float = 0.1,
) -> torch.Tensor:
    """
    Compute routing loss for adaptive processing.

    The routing loss encourages:
    1. Selecting paths that minimize task loss
    2. Preferring computationally cheaper paths

    Args:
        path_weights: Soft path selection weights [batch_size, num_paths]
        task_losses: Per-sample task losses [batch_size]
        path_costs: Cost per path [num_paths]
        cost_weight: Weight for cost regularization

    Returns:
        Routing loss tensor
    """
    # Expected cost based on path selection
    expected_cost = (path_weights * path_costs.unsqueeze(0)).sum(dim=-1)

    # Combined loss: task + cost
    routing_loss = task_losses.mean() + cost_weight * expected_cost.mean()

    return routing_loss


def compute_consistency_loss(
    clean_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Compute consistency regularization loss for robustness.

    Encourages predictions to be stable under perturbations.

    Args:
        clean_logits: Logits from clean input
        perturbed_logits: Logits from perturbed input
        temperature: Temperature for KL divergence

    Returns:
        Consistency loss tensor
    """
    clean_probs = F.softmax(clean_logits / temperature, dim=-1)
    perturbed_log_probs = F.log_softmax(perturbed_logits / temperature, dim=-1)

    # KL divergence
    kl_loss = F.kl_div(
        perturbed_log_probs,
        clean_probs,
        reduction='batchmean'
    )

    return kl_loss


class CaliDocLoss(nn.Module):
    """
    Combined loss function for CaliDoc training.

    Combines task loss, calibration loss, quality loss, and routing loss
    with configurable weights.

    Args:
        task_weight: Weight for task loss
        calibration_weight: Weight for calibration loss
        quality_weight: Weight for quality estimation loss
        routing_weight: Weight for routing loss
        consistency_weight: Weight for consistency loss
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        calibration_weight: float = 0.1,
        quality_weight: float = 0.05,
        routing_weight: float = 0.05,
        consistency_weight: float = 0.0,
        task_type: str = 'classification',
    ):
        super().__init__()

        self.task_weight = task_weight
        self.calibration_weight = calibration_weight
        self.quality_weight = quality_weight
        self.routing_weight = routing_weight
        self.consistency_weight = consistency_weight
        self.task_type = task_type

        # Path costs (relative latency)
        self.register_buffer(
            'path_costs',
            torch.tensor([1.0, 2.5, 1.8])  # Clean, Degraded, Complex
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        path_weights: Optional[torch.Tensor] = None,
        predicted_quality: Optional[torch.Tensor] = None,
        target_quality: Optional[torch.Tensor] = None,
        clean_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        losses = {}

        # Task loss
        task_loss = compute_task_loss(logits, labels, self.task_type)
        losses['task_loss'] = task_loss.item()
        total_loss = self.task_weight * task_loss

        # Calibration loss
        if self.calibration_weight > 0:
            if logits.dim() == 2:  # Classification
                cal_loss = compute_calibration_loss(logits, labels)
                losses['calibration_loss'] = cal_loss.item()
                total_loss += self.calibration_weight * cal_loss

        # Quality loss
        if self.quality_weight > 0 and predicted_quality is not None:
            if target_quality is not None:
                quality_loss = compute_quality_loss(predicted_quality, target_quality)
                losses['quality_loss'] = quality_loss.item()
                total_loss += self.quality_weight * quality_loss

        # Routing loss
        if self.routing_weight > 0 and path_weights is not None:
            # Compute per-sample task loss for routing
            per_sample_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='none'
            )
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.view(logits.size(0), -1).mean(dim=-1)

            routing_loss = compute_routing_loss(
                path_weights, per_sample_loss, self.path_costs
            )
            losses['routing_loss'] = routing_loss.item()
            total_loss += self.routing_weight * routing_loss

        # Consistency loss
        if self.consistency_weight > 0 and clean_logits is not None:
            consistency_loss = compute_consistency_loss(clean_logits, logits)
            losses['consistency_loss'] = consistency_loss.item()
            total_loss += self.consistency_weight * consistency_loss

        losses['total_loss'] = total_loss.item()

        return total_loss, losses
