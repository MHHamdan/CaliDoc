"""
CaliDoc Evaluation Module

Provides evaluation utilities including:
- Task-specific metrics (ANLS, F1)
- Calibration metrics (ECE, MCE, coverage)
- Robustness evaluation
- Conformal prediction analysis
"""

from .metrics import (
    compute_metrics,
    compute_anls,
    compute_ner_f1,
    compute_ece,
    compute_coverage,
    compute_set_size,
)
from .conformal_metrics import (
    evaluate_conformal_predictions,
    compute_conditional_coverage,
    plot_reliability_diagram,
)
from .robustness_eval import (
    evaluate_robustness,
    compute_robustness_gap,
    RobustnessEvaluator,
)

__all__ = [
    # Basic metrics
    "compute_metrics",
    "compute_anls",
    "compute_ner_f1",
    "compute_ece",
    # Conformal metrics
    "compute_coverage",
    "compute_set_size",
    "evaluate_conformal_predictions",
    "compute_conditional_coverage",
    "plot_reliability_diagram",
    # Robustness
    "evaluate_robustness",
    "compute_robustness_gap",
    "RobustnessEvaluator",
]
