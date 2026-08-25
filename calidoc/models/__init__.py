"""
CaliDoc Model Components

This module provides the core model components:
- Document encoders (LayoutLMv3, DocOwl, GOT-OCR)
- Conformal calibration module
- Quality estimator
- Adaptive router
- Full CaliDoc model
"""

from .conformal_calibrator import (
    ConformalCalibrator,
    AdaptiveConformalCalibrator,
    CalibrationResult,
    PredictionResult,
    compute_ece,
    compute_mce,
)
from .quality_estimator import QualityEstimator, QualityScore
from .adaptive_router import AdaptiveRouter, ProcessingPath
from .document_encoder import (
    DocumentEncoder,
    LayoutLMv3Encoder,
    DocOwlEncoder,
    GOTOCREncoder,
)
from .calidoc_model import CaliDocModel

__all__ = [
    # Conformal prediction
    "ConformalCalibrator",
    "AdaptiveConformalCalibrator",
    "CalibrationResult",
    "PredictionResult",
    "compute_ece",
    "compute_mce",
    # Quality estimation
    "QualityEstimator",
    "QualityScore",
    # Routing
    "AdaptiveRouter",
    "ProcessingPath",
    # Document encoders
    "DocumentEncoder",
    "LayoutLMv3Encoder",
    "DocOwlEncoder",
    "GOTOCREncoder",
    # Full model
    "CaliDocModel",
]
