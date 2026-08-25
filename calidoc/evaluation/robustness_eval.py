"""
Robustness Evaluation for Document Understanding

Provides utilities for:
- Systematic perturbation evaluation
- Robustness gap computation
- Error attribution across perturbations
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd

# Lazy import for perturbations (requires cv2/albumentations)
DocumentPerturbations = None

def _get_perturbations():
    global DocumentPerturbations
    if DocumentPerturbations is None:
        from ..data.perturbations import DocumentPerturbations as _DP
        DocumentPerturbations = _DP
    return DocumentPerturbations


@dataclass
class RobustnessResult:
    """Container for robustness evaluation results."""
    clean_accuracy: float
    perturbed_accuracy: Dict[str, Dict[int, float]]  # perturbation -> severity -> accuracy
    robustness_gap: Dict[str, float]  # perturbation -> gap
    overall_robustness_gap: float


class RobustnessEvaluator:
    """
    Evaluator for systematic robustness testing.

    Applies perturbations to documents and measures
    performance degradation across severity levels.

    Args:
        model: Model to evaluate
        perturbations: List of perturbation names to apply
        severity_levels: List of severity levels (1, 2, 3)
        device: Device to run evaluation on
    """

    def __init__(
        self,
        model: torch.nn.Module,
        perturbations: Optional[List[str]] = None,
        severity_levels: List[int] = [1, 2, 3],
        device: str = 'cuda',
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.perturbator = _get_perturbations()()
        self.perturbations = perturbations or list(
            self.perturbator.get_all_perturbations().keys()
        )
        self.severity_levels = severity_levels

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        metric_fn: Callable,
    ) -> RobustnessResult:
        """
        Run full robustness evaluation.

        Args:
            dataloader: DataLoader with evaluation data
            metric_fn: Function to compute metric (predictions, labels) -> score

        Returns:
            RobustnessResult with all metrics
        """
        # Evaluate on clean data
        clean_preds, clean_labels = self._evaluate_clean(dataloader)
        clean_accuracy = metric_fn(clean_preds, clean_labels)

        # Evaluate on perturbed data
        perturbed_results = {}

        for perturbation in tqdm(self.perturbations, desc="Perturbations"):
            perturbed_results[perturbation] = {}

            for severity in self.severity_levels:
                preds, labels = self._evaluate_perturbed(
                    dataloader, perturbation, severity
                )
                perturbed_results[perturbation][severity] = metric_fn(preds, labels)

        # Compute robustness gaps
        robustness_gaps = {}
        for perturbation, results in perturbed_results.items():
            # Gap is the difference between clean and worst-case perturbed
            worst_accuracy = min(results.values())
            robustness_gaps[perturbation] = clean_accuracy - worst_accuracy

        overall_gap = np.mean(list(robustness_gaps.values()))

        return RobustnessResult(
            clean_accuracy=clean_accuracy,
            perturbed_accuracy=perturbed_results,
            robustness_gap=robustness_gaps,
            overall_robustness_gap=overall_gap,
        )

    def _evaluate_clean(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> tuple:
        """Evaluate on clean data."""
        all_preds = []
        all_labels = []

        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            outputs = self.model(
                images=batch.get('pixel_values'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                bbox=batch.get('bbox'),
                return_loss=False,
            )

            preds = outputs['logits'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch['labels'].cpu().numpy().tolist())

        return np.array(all_preds), np.array(all_labels)

    def _evaluate_perturbed(
        self,
        dataloader: torch.utils.data.DataLoader,
        perturbation: str,
        severity: int
    ) -> tuple:
        """Evaluate on perturbed data."""
        perturb_fn = getattr(self.perturbator, perturbation)
        all_preds = []
        all_labels = []

        for batch in dataloader:
            # Apply perturbation to images
            if 'pixel_values' in batch:
                perturbed_images = []
                for img in batch['pixel_values']:
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_np = perturb_fn(img_np, severity=severity)
                    img_tensor = torch.from_numpy(img_np).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    perturbed_images.append(img_tensor)
                batch['pixel_values'] = torch.stack(perturbed_images)

            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            outputs = self.model(
                images=batch.get('pixel_values'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                bbox=batch.get('bbox'),
                return_loss=False,
            )

            preds = outputs['logits'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch['labels'].cpu().numpy().tolist())

        return np.array(all_preds), np.array(all_labels)

    def generate_report(
        self,
        result: RobustnessResult,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate robustness evaluation report.

        Args:
            result: RobustnessResult from evaluate()
            save_path: Path to save CSV report

        Returns:
            DataFrame with detailed results
        """
        rows = []

        for perturbation, severity_results in result.perturbed_accuracy.items():
            for severity, accuracy in severity_results.items():
                rows.append({
                    'perturbation': perturbation,
                    'severity': severity,
                    'accuracy': accuracy,
                    'drop': result.clean_accuracy - accuracy,
                    'relative_drop': (result.clean_accuracy - accuracy) / result.clean_accuracy * 100,
                })

        df = pd.DataFrame(rows)

        if save_path:
            df.to_csv(save_path, index=False)

        return df


def evaluate_robustness(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    perturbations: Optional[List[str]] = None,
    severity_levels: List[int] = [1, 2, 3],
    device: str = 'cuda',
) -> RobustnessResult:
    """
    Convenience function for robustness evaluation.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        perturbations: List of perturbations (None for all)
        severity_levels: Severity levels to test
        device: Device to use

    Returns:
        RobustnessResult
    """
    evaluator = RobustnessEvaluator(
        model=model,
        perturbations=perturbations,
        severity_levels=severity_levels,
        device=device,
    )

    def accuracy_metric(preds, labels):
        return np.mean(np.array(preds).flatten() == np.array(labels).flatten())

    return evaluator.evaluate(dataloader, accuracy_metric)


def compute_robustness_gap(
    clean_accuracy: float,
    perturbed_accuracy: float
) -> float:
    """Compute robustness gap (accuracy drop under perturbation)."""
    return clean_accuracy - perturbed_accuracy


def compute_relative_robustness(
    clean_accuracy: float,
    perturbed_accuracy: float
) -> float:
    """Compute relative robustness (% of clean accuracy retained)."""
    if clean_accuracy == 0:
        return 0.0
    return perturbed_accuracy / clean_accuracy


def aggregate_robustness_by_category(
    result: RobustnessResult
) -> Dict[str, float]:
    """
    Aggregate robustness gaps by perturbation category.

    Args:
        result: RobustnessResult

    Returns:
        Dictionary mapping category to average robustness gap
    """
    categories = _get_perturbations().get_perturbations_by_category()

    category_gaps = {}
    for category, perturbations in categories.items():
        gaps = [
            result.robustness_gap.get(p, 0)
            for p in perturbations
            if p in result.robustness_gap
        ]
        if gaps:
            category_gaps[category] = np.mean(gaps)

    return category_gaps
