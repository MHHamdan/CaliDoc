"""
Adaptive Router Module for Quality-Aware Document Processing

This module implements the adaptive routing mechanism that directs documents
to appropriate processing paths based on quality, complexity, and degradation.

Processing Paths:
- Path A (Clean): Direct VLM processing for high-quality documents
- Path B (Degraded): Enhancement pipeline before VLM for degraded documents
- Path C (Complex): OCR+VLM ensemble for complex structured documents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ProcessingPath(Enum):
    """Available processing paths."""
    CLEAN = "clean"         # Path A: Direct VLM
    DEGRADED = "degraded"   # Path B: Enhancement → VLM
    COMPLEX = "complex"     # Path C: OCR + VLM Ensemble


@dataclass
class RoutingDecision:
    """Container for routing decision outputs."""
    path: ProcessingPath
    confidence: float
    path_probabilities: Dict[ProcessingPath, float]
    quality_score: float
    complexity_score: float


class AdaptiveRouter(nn.Module):
    """
    Adaptive router for quality-aware document processing.

    The router decides which processing path to use based on:
    - Document quality score
    - OCR difficulty estimate
    - Document complexity
    - Degradation type and severity

    Args:
        input_dim: Dimension of quality estimator output
        hidden_dim: Hidden dimension for routing MLP
        num_paths: Number of processing paths (default: 3)
        temperature: Softmax temperature for path selection
        use_gumbel: Whether to use Gumbel-Softmax for differentiable routing
    """

    def __init__(
        self,
        input_dim: int = 3,  # quality, ocr_difficulty, complexity
        hidden_dim: int = 64,
        num_paths: int = 3,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        threshold_quality_high: float = 0.8,
        threshold_quality_low: float = 0.4,
        threshold_complexity_low: float = 0.3,
    ):
        super().__init__()

        self.num_paths = num_paths
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Thresholds for rule-based routing (used during inference)
        self.threshold_quality_high = threshold_quality_high
        self.threshold_quality_low = threshold_quality_low
        self.threshold_complexity_low = threshold_complexity_low

        # Learnable routing network
        self.router_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_paths)
        )

        # Path-specific cost weights (learnable)
        self.path_costs = nn.Parameter(torch.tensor([1.0, 2.5, 1.8]))

        # Initialize with bias toward clean path
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with bias toward efficient paths."""
        for module in self.router_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_path_logits(
        self,
        quality: torch.Tensor,
        ocr_difficulty: torch.Tensor,
        complexity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logits for each processing path.

        Args:
            quality: Document quality scores [batch_size]
            ocr_difficulty: OCR difficulty estimates [batch_size]
            complexity: Document complexity scores [batch_size]

        Returns:
            Path logits [batch_size, num_paths]
        """
        # Stack inputs
        inputs = torch.stack([quality, ocr_difficulty, complexity], dim=-1)

        # Compute logits
        logits = self.router_mlp(inputs)

        return logits

    def forward(
        self,
        quality: torch.Tensor,
        ocr_difficulty: torch.Tensor,
        complexity: torch.Tensor,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for differentiable routing.

        Args:
            quality: Document quality scores [batch_size]
            ocr_difficulty: OCR difficulty estimates [batch_size]
            complexity: Document complexity scores [batch_size]
            hard: Whether to use hard (one-hot) routing

        Returns:
            Tuple of (path_weights, path_indices)
            - path_weights: Soft or hard path weights [batch_size, num_paths]
            - path_indices: Selected path indices [batch_size]
        """
        logits = self.compute_path_logits(quality, ocr_difficulty, complexity)

        if self.training and self.use_gumbel:
            # Gumbel-Softmax for differentiable sampling
            path_weights = F.gumbel_softmax(
                logits, tau=self.temperature, hard=hard
            )
        else:
            # Regular softmax
            path_weights = F.softmax(logits / self.temperature, dim=-1)

        # Get selected paths
        path_indices = path_weights.argmax(dim=-1)

        return path_weights, path_indices

    def route_rule_based(
        self,
        quality: torch.Tensor,
        ocr_difficulty: torch.Tensor,
        complexity: torch.Tensor,
        degradation_max: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Rule-based routing for interpretable decisions.

        This implements the routing logic described in the paper:
        - High quality + low complexity → Clean path
        - Low quality or high degradation → Degraded path
        - Otherwise → Complex path

        Args:
            quality: Document quality scores [batch_size]
            ocr_difficulty: OCR difficulty estimates [batch_size]
            complexity: Document complexity scores [batch_size]
            degradation_max: Maximum degradation severity [batch_size]

        Returns:
            Path indices [batch_size]
        """
        batch_size = quality.shape[0]
        path_indices = torch.zeros(batch_size, dtype=torch.long, device=quality.device)

        # Path A: Clean (high quality, low complexity)
        clean_mask = (quality > self.threshold_quality_high) & \
                     (complexity < self.threshold_complexity_low)

        # Path B: Degraded (low quality or high degradation)
        if degradation_max is not None:
            degraded_mask = (quality < self.threshold_quality_low) | \
                           (degradation_max > 0.5)
        else:
            degraded_mask = quality < self.threshold_quality_low

        # Path C: Complex (everything else)
        complex_mask = ~clean_mask & ~degraded_mask

        path_indices[clean_mask] = 0      # CLEAN
        path_indices[degraded_mask] = 1   # DEGRADED
        path_indices[complex_mask] = 2    # COMPLEX

        return path_indices

    def route(
        self,
        quality: torch.Tensor,
        ocr_difficulty: torch.Tensor,
        complexity: torch.Tensor,
        use_learned: bool = True
    ) -> List[RoutingDecision]:
        """
        Make routing decisions for a batch of documents.

        Args:
            quality: Document quality scores [batch_size]
            ocr_difficulty: OCR difficulty estimates [batch_size]
            complexity: Document complexity scores [batch_size]
            use_learned: Whether to use learned routing (vs rule-based)

        Returns:
            List of RoutingDecision objects
        """
        self.eval()
        with torch.no_grad():
            if use_learned:
                path_weights, path_indices = self.forward(
                    quality, ocr_difficulty, complexity, hard=False
                )
            else:
                path_indices = self.route_rule_based(
                    quality, ocr_difficulty, complexity
                )
                # Create one-hot weights
                path_weights = F.one_hot(path_indices, num_classes=self.num_paths).float()

            # Convert to routing decisions
            decisions = []
            path_mapping = [ProcessingPath.CLEAN, ProcessingPath.DEGRADED, ProcessingPath.COMPLEX]

            for i in range(quality.shape[0]):
                path_probs = {
                    path_mapping[j]: path_weights[i, j].item()
                    for j in range(self.num_paths)
                }

                decisions.append(RoutingDecision(
                    path=path_mapping[path_indices[i].item()],
                    confidence=path_weights[i, path_indices[i]].item(),
                    path_probabilities=path_probs,
                    quality_score=quality[i].item(),
                    complexity_score=complexity[i].item()
                ))

            return decisions

    def compute_routing_loss(
        self,
        path_weights: torch.Tensor,
        task_losses: torch.Tensor,
        cost_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute routing loss for training.

        The routing loss encourages:
        1. Lower task loss (accuracy)
        2. Lower computational cost (efficiency)

        Args:
            path_weights: Soft path weights [batch_size, num_paths]
            task_losses: Per-sample task losses [batch_size]
            cost_weight: Weight for cost regularization

        Returns:
            Routing loss scalar
        """
        # Weighted cost based on path selection
        expected_cost = (path_weights * self.path_costs.unsqueeze(0)).sum(dim=-1)

        # Combined loss: task performance + cost penalty
        routing_loss = task_losses.mean() + cost_weight * expected_cost.mean()

        return routing_loss

    def get_path_statistics(
        self,
        path_indices: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute statistics about path usage.

        Args:
            path_indices: Selected path indices [batch_size]

        Returns:
            Dictionary with path usage statistics
        """
        total = path_indices.shape[0]
        path_mapping = ['clean', 'degraded', 'complex']

        stats = {}
        for i, name in enumerate(path_mapping):
            count = (path_indices == i).sum().item()
            stats[f'{name}_ratio'] = count / total
            stats[f'{name}_count'] = count

        return stats


class CostAwareRouter(AdaptiveRouter):
    """
    Extended router with explicit cost-performance optimization.

    This router optimizes the Pareto frontier between accuracy and
    computational cost using a differentiable cost model.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_paths: int = 3,
        temperature: float = 1.0,
        path_latencies: List[float] = [1.0, 2.5, 1.8],
        cost_lambda: float = 0.1
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_paths=num_paths,
            temperature=temperature
        )

        # Fixed latency costs per path
        self.register_buffer(
            'path_latencies',
            torch.tensor(path_latencies)
        )

        self.cost_lambda = cost_lambda

        # Accuracy predictor (predicts expected accuracy for each path)
        self.accuracy_predictor = nn.Sequential(
            nn.Linear(input_dim + num_paths, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        quality: torch.Tensor,
        ocr_difficulty: torch.Tensor,
        complexity: torch.Tensor,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with cost-aware routing.

        Returns:
            Tuple of (path_weights, path_indices, expected_cost)
        """
        path_weights, path_indices = super().forward(
            quality, ocr_difficulty, complexity, hard
        )

        # Compute expected cost
        expected_cost = (path_weights * self.path_latencies).sum(dim=-1)

        return path_weights, path_indices, expected_cost

    def compute_pareto_loss(
        self,
        path_weights: torch.Tensor,
        task_losses: torch.Tensor,
        expected_costs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Pareto-optimal loss balancing accuracy and cost.

        Args:
            path_weights: Soft path weights [batch_size, num_paths]
            task_losses: Per-sample task losses [batch_size]
            expected_costs: Expected computational costs [batch_size]

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        accuracy_loss = task_losses.mean()
        cost_loss = expected_costs.mean()

        total_loss = accuracy_loss + self.cost_lambda * cost_loss

        metrics = {
            'accuracy_loss': accuracy_loss.item(),
            'cost_loss': cost_loss.item(),
            'total_loss': total_loss.item(),
            'avg_cost': expected_costs.mean().item()
        }

        return total_loss, metrics
