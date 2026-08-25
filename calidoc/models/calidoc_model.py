"""
CaliDoc: Full Model Implementation

This module implements the complete CaliDoc framework that combines:
1. Document encoder (LayoutLMv3, DocOwl, or GOT-OCR)
2. Quality estimator for document quality assessment
3. Adaptive router for path selection
4. Conformal calibrator for uncertainty quantification

The model provides calibrated predictions with coverage guarantees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

from .document_encoder import (
    DocumentEncoder,
    LayoutLMv3Encoder,
    DocOwlEncoder,
    GOTOCREncoder,
    EncoderOutput,
    create_encoder,
)
from .quality_estimator import QualityEstimator, QualityScore
from .adaptive_router import AdaptiveRouter, ProcessingPath, RoutingDecision
from .conformal_calibrator import (
    ConformalCalibrator,
    CalibrationResult,
    PredictionResult,
)


@dataclass
class CaliDocOutput:
    """Complete output from CaliDoc model."""
    # Primary outputs
    answer: str
    confidence: float
    calibrated_confidence: float
    prediction_set: List[str]
    set_size: int

    # Quality and routing info
    quality_score: float
    ocr_difficulty: float
    complexity: float
    processing_path: ProcessingPath

    # Raw model outputs
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None

    # Coverage guarantee
    coverage_level: float = 0.9  # 1 - alpha


class CaliDocModel(nn.Module):
    """
    CaliDoc: Calibrated Document Understanding Model.

    This is the main model class that combines all CaliDoc components
    for uncertainty-aware document understanding.

    Args:
        encoder_type: Type of document encoder ('layoutlmv3', 'docowl', 'got_ocr')
        encoder_config: Configuration for the document encoder
        quality_estimator_config: Configuration for quality estimator
        router_config: Configuration for adaptive router
        calibration_config: Configuration for conformal calibration
    """

    def __init__(
        self,
        encoder_type: str = 'layoutlmv3',
        encoder_config: Optional[Dict] = None,
        quality_estimator_config: Optional[Dict] = None,
        router_config: Optional[Dict] = None,
        calibration_config: Optional[Dict] = None,
        use_quality_routing: bool = True,
        use_calibration: bool = True,
    ):
        super().__init__()

        encoder_config = encoder_config or {}
        quality_estimator_config = quality_estimator_config or {}
        router_config = router_config or {}
        calibration_config = calibration_config or {}

        # Initialize document encoder
        self.encoder_type = encoder_type
        self.encoder = create_encoder(encoder_type, **encoder_config)

        # Initialize quality estimator
        self.use_quality_routing = use_quality_routing
        if use_quality_routing:
            self.quality_estimator = QualityEstimator(**quality_estimator_config)
            self.router = AdaptiveRouter(**router_config)

        # Initialize conformal calibrator
        self.use_calibration = use_calibration
        if use_calibration:
            self.calibrator = ConformalCalibrator(**calibration_config)

        # Image enhancement module for degraded path
        self.enhancer = None  # Can be initialized with RestoreDocs or similar

        # OCR module for complex path
        self.ocr_module = None  # Can be initialized with GOT-OCR

    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CaliDoc.

        This method handles:
        1. Quality estimation
        2. Adaptive routing
        3. Path-specific processing
        4. Uncertainty estimation

        Args:
            images: Document images [batch_size, 3, H, W]
            input_ids: Tokenized text input
            attention_mask: Attention mask
            bbox: Bounding boxes
            pixel_values: Processed image tensors
            labels: Ground truth labels (for training)
            return_loss: Whether to compute and return loss

        Returns:
            Dictionary with predictions, losses, and auxiliary outputs
        """
        batch_size = images.shape[0]
        device = images.device

        outputs = {}

        # Step 1: Quality estimation
        if self.use_quality_routing:
            quality_outputs = self.quality_estimator(images)
            quality = quality_outputs['quality']
            ocr_difficulty = quality_outputs['ocr_difficulty']
            complexity = quality_outputs['complexity']

            outputs['quality'] = quality
            outputs['ocr_difficulty'] = ocr_difficulty
            outputs['complexity'] = complexity

            # Step 2: Adaptive routing
            path_weights, path_indices = self.router(
                quality, ocr_difficulty, complexity
            )
            outputs['path_weights'] = path_weights
            outputs['path_indices'] = path_indices
        else:
            # Default to clean path
            path_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Step 3: Path-specific processing
        # For simplicity in training, we process all samples through the main encoder
        # Path-specific processing is applied during inference

        encoder_output = self.encoder(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs
        )

        outputs['logits'] = encoder_output.logits
        outputs['hidden_states'] = encoder_output.hidden_states
        outputs['attentions'] = encoder_output.attentions
        outputs['layout_attention'] = encoder_output.layout_attention

        # Include start/end logits for VQA tasks
        if encoder_output.start_logits is not None:
            outputs['start_logits'] = encoder_output.start_logits
            outputs['end_logits'] = encoder_output.end_logits

        # For VQA, use encoder's loss directly if available
        if encoder_output.loss is not None:
            outputs['loss'] = encoder_output.loss
            outputs['task_loss'] = encoder_output.loss
            return outputs

        # Step 4: Compute losses (for NER/classification or when encoder doesn't compute loss)
        if return_loss and labels is not None:
            losses = self._compute_losses(
                logits=encoder_output.logits,
                labels=labels,
                quality_outputs=quality_outputs if self.use_quality_routing else None,
                path_weights=path_weights if self.use_quality_routing else None,
                layout_attention=encoder_output.layout_attention,
            )
            outputs.update(losses)

        return outputs

    def _compute_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        quality_outputs: Optional[Dict] = None,
        path_weights: Optional[torch.Tensor] = None,
        layout_attention: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses."""
        losses = {}

        # Task loss (cross-entropy for classification, extractive for VQA)
        if logits.dim() == 3 and logits.shape[-1] == 2:  # VQA (start/end logits)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            start_loss = F.cross_entropy(start_logits, labels[:, 0])
            end_loss = F.cross_entropy(end_logits, labels[:, 1])
            task_loss = (start_loss + end_loss) / 2
        else:
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        losses['task_loss'] = task_loss

        # Calibration loss (ECE minimization)
        if self.use_calibration:
            probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)
            ece = self._compute_ece_loss(probs, labels.view(-1))
            losses['calibration_loss'] = ece

        # Routing loss
        if path_weights is not None:
            # Per-sample task loss for routing optimization
            if logits.dim() == 2:
                per_sample_loss = F.cross_entropy(
                    logits, labels, reduction='none'
                )
            else:
                per_sample_loss = task_loss.unsqueeze(0).expand(logits.shape[0])

            routing_loss = self.router.compute_routing_loss(
                path_weights, per_sample_loss
            )
            losses['routing_loss'] = routing_loss

        # Total loss
        total_loss = task_loss
        if 'calibration_loss' in losses:
            total_loss = total_loss + 0.1 * losses['calibration_loss']
        if 'routing_loss' in losses:
            total_loss = total_loss + 0.05 * losses['routing_loss']

        losses['loss'] = total_loss

        return losses

    def _compute_ece_loss(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15
    ) -> torch.Tensor:
        """Compute differentiable ECE loss."""
        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == labels).float()

        # Soft binning for differentiability
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.zeros(1, device=probs.device)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & \
                     (confidences <= bin_boundaries[i + 1])

            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                ece += in_bin.float().mean() * (accuracy_in_bin - confidence_in_bin).abs()

        return ece

    def calibrate(
        self,
        calibration_dataloader,
        alpha: float = 0.1
    ) -> CalibrationResult:
        """
        Calibrate the conformal predictor on a calibration set.

        Args:
            calibration_dataloader: DataLoader with calibration data
            alpha: Target miscoverage rate

        Returns:
            CalibrationResult with threshold and statistics
        """
        assert self.use_calibration, "Calibration not enabled"

        self.eval()
        all_logits = []
        all_labels = []
        all_layout_attention = []
        all_ocr_confidence = []

        with torch.no_grad():
            for batch in calibration_dataloader:
                outputs = self.forward(
                    images=batch['images'],
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    bbox=batch.get('bbox'),
                    return_loss=False
                )

                all_logits.append(outputs['logits'])
                all_labels.append(batch['labels'])

                if outputs.get('layout_attention') is not None:
                    all_layout_attention.append(outputs['layout_attention'])

        # Concatenate
        cal_logits = torch.cat(all_logits, dim=0)
        cal_labels = torch.cat(all_labels, dim=0)

        layout_attention = None
        if all_layout_attention:
            layout_attention = torch.cat(all_layout_attention, dim=0)

        # Calibrate
        result = self.calibrator.calibrate(
            cal_logits,
            cal_labels,
            layout_attention=layout_attention,
        )

        return result

    @torch.no_grad()
    def predict(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        questions: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        return_raw: bool = False,
    ) -> List[CaliDocOutput]:
        """
        Generate predictions with uncertainty estimates.

        Args:
            images: Document images (tensor or list of PIL images)
            questions: Questions for VQA (if applicable)
            input_ids: Pre-tokenized input
            attention_mask: Attention mask
            bbox: Bounding boxes
            return_raw: Whether to return raw model outputs

        Returns:
            List of CaliDocOutput with calibrated predictions
        """
        self.eval()

        # Prepare inputs
        if isinstance(images, list):
            # Process PIL images
            if hasattr(self.encoder, 'processor'):
                encoding = self.encoder.prepare_inputs(images, questions or [''] * len(images))
                images = encoding['pixel_values']
                input_ids = encoding.get('input_ids')
                attention_mask = encoding.get('attention_mask')
                bbox = encoding.get('bbox')

        # Forward pass
        outputs = self.forward(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            return_loss=False
        )

        # Get quality and routing info
        if self.use_quality_routing:
            quality = outputs['quality']
            ocr_difficulty = outputs['ocr_difficulty']
            complexity = outputs['complexity']
            path_indices = outputs['path_indices']
        else:
            batch_size = images.shape[0]
            quality = torch.ones(batch_size)
            ocr_difficulty = torch.zeros(batch_size)
            complexity = torch.zeros(batch_size)
            path_indices = torch.zeros(batch_size, dtype=torch.long)

        # Generate prediction sets if calibrated
        if self.use_calibration and self.calibrator.is_calibrated:
            prediction_results = self.calibrator.predict(
                outputs['logits'],
                layout_attention=outputs.get('layout_attention'),
            )
        else:
            # Fallback to top-1 prediction
            probs = F.softmax(outputs['logits'], dim=-1)
            predictions = probs.argmax(dim=-1)
            prediction_results = [
                PredictionResult(
                    prediction=pred.item(),
                    confidence=probs[i, pred].item(),
                    prediction_set=[pred.item()],
                    set_size=1,
                    calibrated_confidence=probs[i, pred].item()
                )
                for i, pred in enumerate(predictions)
            ]

        # Build output objects
        path_mapping = [ProcessingPath.CLEAN, ProcessingPath.DEGRADED, ProcessingPath.COMPLEX]

        results = []
        for i, pred_result in enumerate(prediction_results):
            result = CaliDocOutput(
                answer=str(pred_result.prediction),  # Would be decoded in real usage
                confidence=pred_result.confidence,
                calibrated_confidence=pred_result.calibrated_confidence,
                prediction_set=[str(p) for p in pred_result.prediction_set],
                set_size=pred_result.set_size,
                quality_score=quality[i].item() if torch.is_tensor(quality) else quality,
                ocr_difficulty=ocr_difficulty[i].item() if torch.is_tensor(ocr_difficulty) else ocr_difficulty,
                complexity=complexity[i].item() if torch.is_tensor(complexity) else complexity,
                processing_path=path_mapping[path_indices[i].item()],
                logits=outputs['logits'][i] if return_raw else None,
                hidden_states=outputs['hidden_states'][i] if return_raw and outputs.get('hidden_states') is not None else None,
                coverage_level=1 - self.calibrator.alpha if self.use_calibration else 0.0
            )
            results.append(result)

        return results

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs
    ) -> 'CaliDocModel':
        """
        Load a pretrained CaliDoc model.

        Args:
            model_name_or_path: Path to saved model or HuggingFace model ID

        Returns:
            Loaded CaliDocModel
        """
        # Load configuration
        import os
        import json

        config_path = os.path.join(model_name_or_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        # Merge with kwargs
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load weights
        weights_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu')
            # Use strict=False to handle mismatches (e.g., different task heads)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        return model

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config including encoder config for proper reloading
        encoder_config = {}
        if hasattr(self.encoder, 'task'):
            encoder_config['task'] = self.encoder.task
        if hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'config'):
            encoder_config['num_labels'] = getattr(self.encoder.model.config, 'num_labels', None)
            encoder_config['model_name'] = getattr(self.encoder.model.config, '_name_or_path', None)

        config = {
            'encoder_type': self.encoder_type,
            'encoder_config': encoder_config,
            'use_quality_routing': self.use_quality_routing,
            'use_calibration': self.use_calibration,
        }
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Save weights
        torch.save(
            self.state_dict(),
            os.path.join(save_directory, 'pytorch_model.bin')
        )

        # Save calibration threshold if calibrated
        if self.use_calibration and self.calibrator.is_calibrated:
            torch.save(
                {'threshold': self.calibrator.threshold.item()},
                os.path.join(save_directory, 'calibration.pt')
            )
