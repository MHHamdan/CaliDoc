"""
Quality Estimator Module for Document Understanding

This module implements a lightweight quality estimation network that predicts:
1. Overall document quality score
2. OCR difficulty estimate
3. Document complexity score

The quality estimator enables quality-aware routing for adaptive processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


@dataclass
class QualityScore:
    """Container for quality estimation outputs."""
    quality: float          # Overall quality [0, 1]
    ocr_difficulty: float   # OCR difficulty [0, 1]
    complexity: float       # Document complexity [0, 1]
    degradation_vector: Optional[torch.Tensor] = None  # Per-degradation scores


class QualityEstimator(nn.Module):
    """
    Lightweight quality estimation network for documents.

    Architecture:
    - ResNet-18 backbone (optionally frozen)
    - Multi-head output for quality, OCR difficulty, and complexity
    - Optional degradation classifier

    Args:
        backbone: Backbone architecture ('resnet18', 'resnet34', 'efficientnet_b0')
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone weights
        num_degradation_types: Number of degradation types for classification
        dropout: Dropout rate for classification heads
    """

    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_degradation_types: int = 20,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_degradation_types = num_degradation_types

        # Initialize backbone
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove classification head
        if 'resnet' in backbone:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Quality estimation heads
        self.quality_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.ocr_difficulty_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.complexity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Degradation classifier (multi-label)
        self.degradation_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_degradation_types),
            nn.Sigmoid()
        )

        # Degradation severity regressor
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_degradation_types),
            nn.Sigmoid()  # Severity in [0, 1] for each degradation type
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from document images."""
        features = self.backbone(images)
        if features.dim() > 2:
            features = features.flatten(1)
        return features

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for quality estimation.

        Args:
            images: Document images [batch_size, 3, H, W]
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with quality scores and optional features
        """
        # Extract features
        features = self.extract_features(images)

        # Compute quality scores
        quality = self.quality_head(features).squeeze(-1)
        ocr_difficulty = self.ocr_difficulty_head(features).squeeze(-1)
        complexity = self.complexity_head(features).squeeze(-1)

        # Compute degradation predictions
        degradation_probs = self.degradation_head(features)
        degradation_severity = self.severity_head(features)

        outputs = {
            'quality': quality,
            'ocr_difficulty': ocr_difficulty,
            'complexity': complexity,
            'degradation_probs': degradation_probs,
            'degradation_severity': degradation_severity,
        }

        if return_features:
            outputs['features'] = features

        return outputs

    def predict(self, images: torch.Tensor) -> List[QualityScore]:
        """
        Get quality scores for a batch of images.

        Args:
            images: Document images [batch_size, 3, H, W]

        Returns:
            List of QualityScore objects
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)

            results = []
            batch_size = images.shape[0]

            for i in range(batch_size):
                results.append(QualityScore(
                    quality=outputs['quality'][i].item(),
                    ocr_difficulty=outputs['ocr_difficulty'][i].item(),
                    complexity=outputs['complexity'][i].item(),
                    degradation_vector=outputs['degradation_severity'][i]
                ))

            return results

    def get_routing_scores(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get scores for adaptive routing decisions.

        Returns:
            Tuple of (quality, ocr_difficulty, complexity) tensors
        """
        outputs = self.forward(images)
        return outputs['quality'], outputs['ocr_difficulty'], outputs['complexity']


class QualityEstimatorWithOCR(QualityEstimator):
    """
    Extended quality estimator that uses OCR output as additional signal.

    This version takes both images and OCR outputs (from a preliminary OCR pass)
    to better estimate quality and OCR reliability.
    """

    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        ocr_feature_dim: int = 256,
        num_degradation_types: int = 20,
        dropout: float = 0.2
    ):
        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            num_degradation_types=num_degradation_types,
            dropout=dropout
        )

        # OCR feature encoder
        self.ocr_encoder = nn.Sequential(
            nn.Linear(ocr_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256)
        )

        # Get backbone feature dimension
        if 'resnet' in backbone:
            backbone_dim = 512
        else:
            backbone_dim = 1280

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(backbone_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Update heads to use fused features
        self.quality_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.ocr_difficulty_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        images: torch.Tensor,
        ocr_features: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional OCR features.

        Args:
            images: Document images [batch_size, 3, H, W]
            ocr_features: OCR-derived features [batch_size, ocr_feature_dim]
            return_features: Whether to return intermediate features
        """
        # Extract image features
        image_features = self.extract_features(images)

        if ocr_features is not None:
            # Encode and fuse OCR features
            ocr_encoded = self.ocr_encoder(ocr_features)
            fused = self.fusion(torch.cat([image_features, ocr_encoded], dim=-1))
        else:
            # Use only image features (pad OCR features with zeros)
            dummy_ocr = torch.zeros(images.shape[0], 256, device=images.device)
            fused = self.fusion(torch.cat([image_features, dummy_ocr], dim=-1))

        # Compute quality scores using fused features
        quality = self.quality_head(fused).squeeze(-1)
        ocr_difficulty = self.ocr_difficulty_head(fused).squeeze(-1)
        complexity = self.complexity_head(fused).squeeze(-1)

        outputs = {
            'quality': quality,
            'ocr_difficulty': ocr_difficulty,
            'complexity': complexity,
        }

        if return_features:
            outputs['features'] = fused

        return outputs


def compute_ocr_features(
    ocr_output: Dict,
    max_tokens: int = 512
) -> torch.Tensor:
    """
    Extract features from OCR output for quality estimation.

    Args:
        ocr_output: Dictionary containing OCR results with confidence scores
        max_tokens: Maximum number of tokens to consider

    Returns:
        Feature tensor [feature_dim]
    """
    features = []

    # Average confidence
    if 'confidences' in ocr_output:
        confidences = ocr_output['confidences'][:max_tokens]
        features.extend([
            torch.mean(torch.tensor(confidences)),
            torch.std(torch.tensor(confidences)),
            torch.min(torch.tensor(confidences)),
            torch.quantile(torch.tensor(confidences), 0.1),
        ])

    # Text statistics
    if 'text' in ocr_output:
        text = ocr_output['text']
        features.extend([
            len(text),  # Total length
            text.count(' '),  # Word count proxy
            len([c for c in text if c.isupper()]) / max(len(text), 1),  # Uppercase ratio
            len([c for c in text if c.isdigit()]) / max(len(text), 1),  # Digit ratio
        ])

    # Bounding box statistics
    if 'boxes' in ocr_output:
        boxes = ocr_output['boxes']
        if len(boxes) > 0:
            widths = [b[2] - b[0] for b in boxes]
            heights = [b[3] - b[1] for b in boxes]
            features.extend([
                np.mean(widths),
                np.std(widths),
                np.mean(heights),
                np.std(heights),
            ])

    # Pad to fixed size
    while len(features) < 256:
        features.append(0.0)

    return torch.tensor(features[:256], dtype=torch.float32)
