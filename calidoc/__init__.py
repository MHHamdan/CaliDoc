"""
CaliDoc: Calibrated Document Understanding with Uncertainty Quantification

A framework for uncertainty-aware document understanding that provides:
1. Conformal Prediction for Document VLMs
2. DocRobust-Bench: Comprehensive robustness benchmark
3. Quality-Aware Adaptive Processing

Usage:
    from calidoc import CaliDocModel, ConformalCalibrator

    model = CaliDocModel.from_pretrained("calidoc-layoutlmv3-large")
    result = model.predict(image="document.png", question="What is the total?")
"""

__version__ = "0.1.0"
__author__ = "CaliDoc Team"

from .models import (
    CaliDocModel,
    ConformalCalibrator,
    QualityEstimator,
    AdaptiveRouter,
)

__all__ = [
    "CaliDocModel",
    "ConformalCalibrator",
    "QualityEstimator",
    "AdaptiveRouter",
    "__version__",
]
