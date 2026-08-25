"""
Conformal Calibration Module for Document Understanding

This module implements structure-aware conformal prediction specifically designed
for document understanding tasks. It provides calibrated prediction sets with
provable coverage guarantees.

Key Features:
- Structure-aware nonconformity scores using layout attention and OCR confidence
- Multiple score functions: LAC, APS, and novel Structure-Aware score
- Calibration on held-out set with coverage verification
- Support for both classification and generation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    threshold: float
    empirical_coverage: float
    avg_set_size: float
    score_distribution: np.ndarray


@dataclass
class PredictionResult:
    """Container for prediction results with uncertainty."""
    prediction: int
    confidence: float
    prediction_set: List[int]
    set_size: int
    calibrated_confidence: float
    

class ConformalCalibrator(nn.Module):
    """
    Structure-aware conformal prediction for document understanding.
    
    This class implements conformal prediction with document-specific
    nonconformity scores that incorporate:
    1. Model softmax probabilities
    2. Layout attention entropy (how focused is the model on specific regions)
    3. OCR confidence (reliability of text extraction)
    
    Args:
        alpha: Target miscoverage rate (1-alpha = coverage guarantee)
        score_type: Type of nonconformity score ('lac', 'aps', 'structure_aware')
        lambda_layout: Weight for layout entropy term in structure-aware score
        lambda_ocr: Weight for OCR confidence term in structure-aware score
        temperature: Initial temperature for scaling (learned during calibration)
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        score_type: str = 'structure_aware',
        lambda_layout: float = 0.1,
        lambda_ocr: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        assert 0 < alpha < 1, "Alpha must be between 0 and 1"
        assert score_type in ['lac', 'aps', 'structure_aware'], \
            f"Unknown score type: {score_type}"
        
        self.alpha = alpha
        self.score_type = score_type
        self.lambda_layout = lambda_layout
        self.lambda_ocr = lambda_ocr
        
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Calibration threshold (set during calibration)
        self.register_buffer('threshold', torch.tensor(float('inf')))
        self.is_calibrated = False
        
        # Store calibration statistics
        self.calibration_scores = None
        
    def compute_nonconformity_score(
        self,
        logits: torch.Tensor,
        true_labels: torch.Tensor,
        layout_attention: Optional[torch.Tensor] = None,
        ocr_confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute nonconformity scores for calibration or prediction.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            true_labels: Ground truth labels [batch_size]
            layout_attention: Attention weights over layout regions [batch_size, num_regions]
            ocr_confidence: OCR confidence scores [batch_size] or [batch_size, num_tokens]
            
        Returns:
            Nonconformity scores [batch_size]
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        if self.score_type == 'lac':
            # Least Ambiguous Class: 1 - probability of true class
            true_probs = probs.gather(1, true_labels.unsqueeze(1)).squeeze(1)
            scores = 1 - true_probs
            
        elif self.score_type == 'aps':
            # Adaptive Prediction Sets: cumulative probability until true class
            sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
            
            # Find position of true class in sorted order
            true_class_positions = (sorted_indices == true_labels.unsqueeze(1)).float()
            cumsum_probs = sorted_probs.cumsum(dim=-1)
            
            # Score is cumsum at position of true class
            scores = (cumsum_probs * true_class_positions).sum(dim=-1)
            
        elif self.score_type == 'structure_aware':
            # Novel structure-aware score
            true_probs = probs.gather(1, true_labels.unsqueeze(1)).squeeze(1)
            base_score = 1 - true_probs
            
            # Add layout entropy penalty
            if layout_attention is not None:
                # Compute entropy of attention distribution
                layout_entropy = self._compute_entropy(layout_attention)
                base_score = base_score + self.lambda_layout * layout_entropy
            
            # Add OCR uncertainty penalty
            if ocr_confidence is not None:
                # Handle both per-sample and per-token confidence
                if ocr_confidence.dim() > 1:
                    ocr_conf = ocr_confidence.mean(dim=-1)
                else:
                    ocr_conf = ocr_confidence
                base_score = base_score + self.lambda_ocr * (1 - ocr_conf)
            
            scores = base_score
        
        return scores
    
    def _compute_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distribution."""
        # Normalize if not already
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Compute entropy: -sum(p * log(p))
        log_attention = torch.log(attention + 1e-10)
        entropy = -(attention * log_attention).sum(dim=-1)
        
        # Normalize by max entropy
        max_entropy = np.log(attention.shape[-1])
        return entropy / max_entropy
    
    def calibrate(
        self,
        cal_logits: torch.Tensor,
        cal_labels: torch.Tensor,
        layout_attention: Optional[torch.Tensor] = None,
        ocr_confidence: Optional[torch.Tensor] = None,
        optimize_temperature: bool = True
    ) -> CalibrationResult:
        """
        Calibrate the conformal predictor on a held-out calibration set.
        
        This sets the threshold such that the prediction sets achieve
        (1-alpha) coverage on the calibration set, which transfers to
        the test set under exchangeability.
        
        Args:
            cal_logits: Calibration set logits [n_cal, num_classes]
            cal_labels: Calibration set labels [n_cal]
            layout_attention: Optional attention weights [n_cal, num_regions]
            ocr_confidence: Optional OCR confidence [n_cal]
            optimize_temperature: Whether to optimize temperature first
            
        Returns:
            CalibrationResult with threshold and statistics
        """
        self.eval()
        
        with torch.no_grad():
            # Optionally optimize temperature first (for better efficiency)
            if optimize_temperature:
                self._optimize_temperature(cal_logits, cal_labels)
            
            # Compute nonconformity scores on calibration set
            scores = self.compute_nonconformity_score(
                cal_logits, cal_labels, layout_attention, ocr_confidence
            )
            
            self.calibration_scores = scores.cpu().numpy()
            
            # Compute threshold: (1-alpha) quantile with finite sample correction
            n = len(scores)
            quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            quantile_level = min(quantile_level, 1.0)
            
            threshold = torch.quantile(scores, quantile_level).item()
            self.threshold.fill_(threshold)
            self.is_calibrated = True
            
            # Compute calibration statistics
            empirical_coverage = (scores <= threshold).float().mean().item()
            
            # Compute average set size (for efficiency metric)
            avg_set_size = self._compute_avg_set_size(cal_logits, layout_attention, ocr_confidence)
        
        return CalibrationResult(
            threshold=threshold,
            empirical_coverage=empirical_coverage,
            avg_set_size=avg_set_size,
            score_distribution=self.calibration_scores
        )
    
    def _optimize_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50
    ):
        """Optimize temperature using NLL on calibration set."""
        self.temperature.requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        self.temperature.requires_grad_(False)
    
    def _compute_avg_set_size(
        self,
        logits: torch.Tensor,
        layout_attention: Optional[torch.Tensor] = None,
        ocr_confidence: Optional[torch.Tensor] = None
    ) -> float:
        """Compute average prediction set size."""
        n_classes = logits.shape[-1]
        batch_size = logits.shape[0]
        
        total_set_size = 0
        for i in range(batch_size):
            sample_logits = logits[i:i+1]
            sample_layout = layout_attention[i:i+1] if layout_attention is not None else None
            sample_ocr = ocr_confidence[i:i+1] if ocr_confidence is not None else None
            
            set_size = 0
            for c in range(n_classes):
                dummy_label = torch.tensor([c], device=logits.device)
                score = self.compute_nonconformity_score(
                    sample_logits, dummy_label, sample_layout, sample_ocr
                )
                if score.item() <= self.threshold.item():
                    set_size += 1
            total_set_size += set_size
        
        return total_set_size / batch_size
    
    def predict(
        self,
        logits: torch.Tensor,
        layout_attention: Optional[torch.Tensor] = None,
        ocr_confidence: Optional[torch.Tensor] = None,
        return_all_scores: bool = False
    ) -> List[PredictionResult]:
        """
        Generate prediction sets with coverage guarantee.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            layout_attention: Optional attention weights
            ocr_confidence: Optional OCR confidence
            return_all_scores: Whether to return scores for all classes
            
        Returns:
            List of PredictionResult for each sample
        """
        assert self.is_calibrated, "Must call calibrate() before predict()"
        
        self.eval()
        with torch.no_grad():
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            batch_size, n_classes = logits.shape
            results = []
            
            for i in range(batch_size):
                sample_logits = logits[i:i+1]
                sample_probs = probs[i]
                sample_layout = layout_attention[i:i+1] if layout_attention is not None else None
                sample_ocr = ocr_confidence[i:i+1] if ocr_confidence is not None else None
                
                # Find prediction set
                prediction_set = []
                class_scores = []
                
                for c in range(n_classes):
                    dummy_label = torch.tensor([c], device=logits.device)
                    score = self.compute_nonconformity_score(
                        sample_logits, dummy_label, sample_layout, sample_ocr
                    )
                    class_scores.append(score.item())
                    
                    if score.item() <= self.threshold.item():
                        prediction_set.append(c)
                
                # Get argmax prediction and its confidence
                prediction = sample_probs.argmax().item()
                confidence = sample_probs[prediction].item()
                
                # Compute calibrated confidence
                # If prediction is in set, use 1 - alpha; otherwise use probability-based
                if prediction in prediction_set:
                    calibrated_conf = 1 - self.alpha
                else:
                    calibrated_conf = confidence * (1 - self.alpha)
                
                results.append(PredictionResult(
                    prediction=prediction,
                    confidence=confidence,
                    prediction_set=prediction_set,
                    set_size=len(prediction_set),
                    calibrated_confidence=calibrated_conf
                ))
        
        return results
    
    def compute_coverage(
        self,
        logits: torch.Tensor,
        true_labels: torch.Tensor,
        layout_attention: Optional[torch.Tensor] = None,
        ocr_confidence: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute empirical coverage and related metrics on test data.
        
        Returns:
            Dictionary with coverage, avg_set_size, and other metrics
        """
        results = self.predict(logits, layout_attention, ocr_confidence)
        
        covered = 0
        total_set_size = 0
        empty_sets = 0
        
        for i, result in enumerate(results):
            if true_labels[i].item() in result.prediction_set:
                covered += 1
            total_set_size += result.set_size
            if result.set_size == 0:
                empty_sets += 1
        
        n = len(results)
        return {
            'coverage': covered / n,
            'avg_set_size': total_set_size / n,
            'empty_set_rate': empty_sets / n,
            'coverage_gap': (covered / n) - (1 - self.alpha)
        }


class AdaptiveConformalCalibrator(ConformalCalibrator):
    """
    Adaptive conformal calibrator with sample-dependent thresholds.
    
    This extends the base calibrator to allow different thresholds
    based on document characteristics (quality, complexity).
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        score_type: str = 'structure_aware',
        lambda_layout: float = 0.1,
        lambda_ocr: float = 0.1,
        num_bins: int = 5
    ):
        super().__init__(alpha, score_type, lambda_layout, lambda_ocr)
        self.num_bins = num_bins
        self.bin_thresholds = nn.Parameter(torch.zeros(num_bins))
    
    def calibrate_adaptive(
        self,
        cal_logits: torch.Tensor,
        cal_labels: torch.Tensor,
        quality_scores: torch.Tensor,
        **kwargs
    ) -> List[float]:
        """
        Calibrate with quality-stratified thresholds.
        
        This provides tighter prediction sets for high-quality documents
        and wider sets for low-quality documents.
        """
        # Bin samples by quality
        bin_edges = torch.quantile(quality_scores, 
                                   torch.linspace(0, 1, self.num_bins + 1))
        
        thresholds = []
        for b in range(self.num_bins):
            mask = (quality_scores >= bin_edges[b]) & (quality_scores < bin_edges[b + 1])
            if mask.sum() > 0:
                bin_result = self.calibrate(
                    cal_logits[mask], cal_labels[mask], **kwargs
                )
                thresholds.append(bin_result.threshold)
            else:
                thresholds.append(float('inf'))
        
        self.bin_thresholds.data = torch.tensor(thresholds)
        return thresholds


# Utility functions for ECE computation
def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        probs: Predicted probabilities [batch_size, num_classes]
        labels: True labels [batch_size]
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better)
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * abs(accuracy_in_bin - confidence_in_bin)
    
    return ece.item()


def compute_mce(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> float:
    """Compute Maximum Calibration Error."""
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    max_error = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            confidence_in_bin = confidences[in_bin].mean()
            max_error = max(max_error, abs(accuracy_in_bin - confidence_in_bin).item())
    
    return max_error
