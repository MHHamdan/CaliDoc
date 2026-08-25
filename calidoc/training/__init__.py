"""
CaliDoc Training Module

Provides training utilities including:
- Trainer class with multi-task learning
- Loss functions (task, calibration, routing)
- Optimizers and schedulers
"""

from .trainer import CaliDocTrainer
from .losses import (
    compute_task_loss,
    compute_calibration_loss,
    compute_routing_loss,
    CaliDocLoss,
)

__all__ = [
    "CaliDocTrainer",
    "compute_task_loss",
    "compute_calibration_loss",
    "compute_routing_loss",
    "CaliDocLoss",
]
