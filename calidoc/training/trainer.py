"""
CaliDoc Trainer

Main training class for CaliDoc models with support for:
- Multi-task learning
- Mixed precision training
- Gradient accumulation
- Curriculum learning for robustness
- WandB logging
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Lazy import for wandb (optional dependency)
wandb = None

def _get_wandb():
    global wandb
    if wandb is None:
        import wandb as _wandb
        wandb = _wandb
    return wandb

from .losses import CaliDocLoss, compute_task_loss_with_focal
from ..models import CaliDocModel


logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a smoothed version of model weights which typically
    generalizes better than the raw weights, especially on small datasets.

    Usage:
        ema = EMA(model, decay=0.999)
        for batch in dataloader:
            loss.backward()
            optimizer.step()
            ema.update()  # Update EMA weights

        # For evaluation, use EMA weights
        ema.apply_shadow()  # Apply EMA weights
        evaluate(model)
        ema.restore()  # Restore original weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        """Register model parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update EMA weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


@dataclass
class TrainingConfig:
    """Configuration for CaliDoc training."""
    # Optimization
    epochs: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Loss weights
    task_weight: float = 1.0
    calibration_weight: float = 0.1
    quality_weight: float = 0.05
    routing_weight: float = 0.05
    consistency_weight: float = 0.0

    # Checkpointing
    save_strategy: str = 'epoch'
    save_total_limit: int = 3
    output_dir: str = './outputs'

    # Hardware
    mixed_precision: bool = True
    device: str = 'cuda'

    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = True
    wandb_project: str = 'calidoc'

    # Robustness
    apply_augmentation: bool = True
    curriculum_enabled: bool = False
    curriculum_start_severity: int = 1
    curriculum_end_severity: int = 3
    curriculum_warmup_epochs: int = 5

    # Task-specific settings
    task: str = 'ner'  # 'ner', 'vqa', 'classification'
    id2label: Optional[Dict] = field(default=None)  # Label mapping for NER evaluation

    # Early stopping on the eval dataloader (Phase 2: val_es / es_holdout ONLY).
    # 0 disables; N stops after N consecutive epochs without eval_f1 improvement.
    early_stopping_patience: int = 0

    # Advanced optimization
    llrd_rate: float = 1.0  # Layer-wise LR decay rate (0.95 = 5% decay per layer, 1.0 = disabled)

    # EMA (Exponential Moving Average) - key for small datasets
    use_ema: bool = False
    ema_decay: float = 0.999  # EMA decay rate (0.999 = slow update, 0.99 = faster)

    # SWA (Stochastic Weight Averaging)
    use_swa: bool = False
    swa_start_epoch: int = 50  # Start SWA after this epoch
    swa_lr: float = 1e-5  # SWA learning rate


class CaliDocTrainer:
    """
    Trainer for CaliDoc models.

    Handles the full training loop including:
    - Forward/backward passes
    - Optimization
    - Evaluation
    - Checkpointing
    - Logging

    Args:
        model: CaliDoc model to train
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        config: Training configuration
    """

    def __init__(
        self,
        model: CaliDocModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()

        # Move model to device
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Initialize loss function
        self.loss_fn = CaliDocLoss(
            task_weight=self.config.task_weight,
            calibration_weight=self.config.calibration_weight,
            quality_weight=self.config.quality_weight,
            routing_weight=self.config.routing_weight,
            consistency_weight=self.config.consistency_weight,
        )

        # Store label smoothing and focal loss settings
        self.label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
        self.use_focal_loss = getattr(self.config, 'use_focal_loss', False)
        self.focal_gamma = getattr(self.config, 'focal_gamma', 2.0)
        if self.label_smoothing > 0:
            logger.info(f"Using label smoothing: {self.label_smoothing}")
        if self.use_focal_loss:
            logger.info(f"Using focal loss with gamma={self.focal_gamma}")

        # Mixed precision training
        self.scaler = GradScaler() if self.config.mixed_precision else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None

        # EMA (Exponential Moving Average)
        self.ema = None
        if getattr(self.config, 'use_ema', False):
            ema_decay = getattr(self.config, 'ema_decay', 0.999)
            self.ema = EMA(self.model, decay=ema_decay)
            logger.info(f"Using EMA with decay={ema_decay}")

        # SWA (Stochastic Weight Averaging)
        self.swa_model = None
        self.swa_scheduler = None
        if getattr(self.config, 'use_swa', False):
            from torch.optim.swa_utils import AveragedModel, SWALR
            self.swa_model = AveragedModel(self.model)
            swa_lr = getattr(self.config, 'swa_lr', 1e-5)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=swa_lr)
            logger.info(f"Using SWA starting at epoch {getattr(self.config, 'swa_start_epoch', 50)}")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Initialize WandB (lazy import)
        if self.config.use_wandb:
            _get_wandb().init(
                project=self.config.wandb_project,
                config=vars(self.config)
            )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay and optional layer-wise LR decay (LLRD)."""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

        # Layer-wise learning rate decay (LLRD) - crucial for fine-tuning transformers
        llrd_rate = getattr(self.config, 'llrd_rate', 1.0)  # Default: no LLRD

        if llrd_rate < 1.0:
            logger.info(f"Using layer-wise LR decay with rate {llrd_rate}")
            optimizer_grouped_parameters = self._get_llrd_params(no_decay, llrd_rate)
        else:
            # Standard optimizer without LLRD
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters()
                              if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config.weight_decay,
                },
                {
                    'params': [p for n, p in self.model.named_parameters()
                              if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                },
            ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _get_llrd_params(self, no_decay: List[str], llrd_rate: float) -> List[Dict]:
        """Get optimizer parameter groups with layer-wise learning rate decay."""
        optimizer_grouped_parameters = []

        # Auto-detect number of layers from model parameters
        num_layers = 24  # Default for LayoutLMv3-Large
        for name, _ in self.model.named_parameters():
            # Look for layer indices in parameter names
            if 'layer.' in name or 'layers.' in name:
                import re
                matches = re.findall(r'layer\.(\d+)|layers\.(\d+)', name)
                for match in matches:
                    layer_idx = int(match[0] or match[1])
                    num_layers = max(num_layers, layer_idx + 1)
        logger.info(f"Detected {num_layers} transformer layers for LLRD")

        # Embeddings get lowest LR
        embed_params_decay = [p for n, p in self.model.named_parameters()
                             if 'embed' in n.lower() and not any(nd in n for nd in no_decay)]
        embed_params_no_decay = [p for n, p in self.model.named_parameters()
                                 if 'embed' in n.lower() and any(nd in n for nd in no_decay)]
        embed_lr = self.config.learning_rate * (llrd_rate ** num_layers)

        if embed_params_decay:
            optimizer_grouped_parameters.append({
                'params': embed_params_decay,
                'weight_decay': self.config.weight_decay,
                'lr': embed_lr,
            })
        if embed_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': embed_params_no_decay,
                'weight_decay': 0.0,
                'lr': embed_lr,
            })

        # Transformer layers with LLRD
        for layer_idx in range(num_layers):
            layer_params = [(n, p) for n, p in self.model.named_parameters()
                           if f'layer.{layer_idx}.' in n or f'layers.{layer_idx}.' in n]
            if not layer_params:
                continue

            layer_lr = self.config.learning_rate * (llrd_rate ** (num_layers - layer_idx - 1))
            layer_decay = [p for n, p in layer_params if not any(nd in n for nd in no_decay)]
            layer_no_decay = [p for n, p in layer_params if any(nd in n for nd in no_decay)]

            if layer_decay:
                optimizer_grouped_parameters.append({
                    'params': layer_decay,
                    'weight_decay': self.config.weight_decay,
                    'lr': layer_lr,
                })
            if layer_no_decay:
                optimizer_grouped_parameters.append({
                    'params': layer_no_decay,
                    'weight_decay': 0.0,
                    'lr': layer_lr,
                })

        # Classifier head gets full LR
        head_params_decay = [p for n, p in self.model.named_parameters()
                           if ('classifier' in n.lower() or 'head' in n.lower() or 'qa_output' in n.lower())
                           and not any(nd in n for nd in no_decay)]
        head_params_no_decay = [p for n, p in self.model.named_parameters()
                               if ('classifier' in n.lower() or 'head' in n.lower() or 'qa_output' in n.lower())
                               and any(nd in n for nd in no_decay)]

        if head_params_decay:
            optimizer_grouped_parameters.append({
                'params': head_params_decay,
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate,
            })
        if head_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': head_params_no_decay,
                'weight_decay': 0.0,
                'lr': self.config.learning_rate,
            })

        # Any remaining parameters (full LR)
        assigned_params = set()
        for group in optimizer_grouped_parameters:
            for p in group['params']:
                assigned_params.add(id(p))

        remaining_decay = [p for n, p in self.model.named_parameters()
                         if id(p) not in assigned_params and not any(nd in n for nd in no_decay)]
        remaining_no_decay = [p for n, p in self.model.named_parameters()
                             if id(p) not in assigned_params and any(nd in n for nd in no_decay)]

        if remaining_decay:
            optimizer_grouped_parameters.append({
                'params': remaining_decay,
                'weight_decay': self.config.weight_decay,
            })
        if remaining_no_decay:
            optimizer_grouped_parameters.append({
                'params': remaining_no_decay,
                'weight_decay': 0.0,
            })

        return optimizer_grouped_parameters

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        from transformers import get_cosine_schedule_with_warmup

        num_training_steps = (
            len(self.train_dataloader) *
            self.config.epochs //
            self.config.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.

        Returns:
            Dictionary with final training metrics
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Total training steps: {len(self.train_dataloader) * self.config.epochs}")

        best_metrics = {}
        epochs_without_improvement = 0

        start_epoch = getattr(self, '_resume_epoch', 0)
        if start_epoch:
            logger.info(f"Resuming training at epoch {start_epoch}")

        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Get current curriculum severity
            if self.config.curriculum_enabled:
                severity = self._get_curriculum_severity(epoch)
                logger.info(f"Epoch {epoch}: Using severity {severity}")

            # Training epoch
            train_metrics = self._train_epoch()

            # Update SWA model if active
            swa_start = getattr(self.config, 'swa_start_epoch', 50)
            if self.swa_model is not None and epoch >= swa_start:
                self.swa_model.update_parameters(self.model)

            # Evaluation - use EMA weights if available
            if self.eval_dataloader is not None:
                if self.ema is not None:
                    self.ema.apply_shadow()  # Apply EMA weights

                eval_metrics = self._evaluate()

                if self.ema is not None:
                    self.ema.restore()  # Restore original weights

                # Check if best model
                primary_metric = eval_metrics.get('eval_f1', eval_metrics.get('eval_accuracy', 0))
                if self.best_metric is None or primary_metric > self.best_metric:
                    self.best_metric = primary_metric
                    best_metrics = eval_metrics
                    epochs_without_improvement = 0
                    # Save best with EMA weights if available
                    if self.ema is not None:
                        self.ema.apply_shadow()
                    self._save_checkpoint('best')
                    if self.ema is not None:
                        self.ema.restore()
                else:
                    epochs_without_improvement += 1

            # Save checkpoint; 'latest' is overwritten every epoch for SLURM
            # requeue resume.
            if self.config.save_strategy == 'epoch':
                self._save_checkpoint(f'epoch_{epoch}')
            self._save_checkpoint('latest')

            # Log metrics
            metrics = {**train_metrics, **eval_metrics} if self.eval_dataloader else train_metrics
            if self.config.use_wandb:
                _get_wandb().log(metrics, step=self.global_step)

            logger.info(f"Epoch {epoch}: {metrics}")

            patience = getattr(self.config, 'early_stopping_patience', 0)
            if patience and epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch}: no eval improvement for "
                    f"{epochs_without_improvement} epochs (patience={patience})"
                )
                break

        # Update SWA batch normalization statistics
        if self.swa_model is not None:
            logger.info("Updating SWA batch normalization statistics...")
            from torch.optim.swa_utils import update_bn
            update_bn(self.train_dataloader, self.swa_model, device=self.device)

        # Final save with EMA weights if available
        if self.ema is not None:
            self.ema.apply_shadow()
        self._save_checkpoint('final')
        if self.ema is not None:
            self.ema.restore()

        if self.config.use_wandb:
            _get_wandb().finish()

        return best_metrics

    def _train_epoch(self) -> Dict[str, float]:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                # Determine if we need custom loss (label smoothing or focal loss)
                use_custom_loss = self.label_smoothing > 0 or self.use_focal_loss

                outputs = self.model(
                    images=batch.get('pixel_values'),
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    bbox=batch.get('bbox'),
                    labels=batch.get('labels') if not use_custom_loss else None,
                    start_positions=batch.get('start_positions'),
                    end_positions=batch.get('end_positions'),
                    return_loss=not use_custom_loss,
                )

                # Compute loss with label smoothing or focal loss if enabled
                if use_custom_loss and 'logits' in outputs:
                    labels = batch.get('labels')
                    if labels is not None:
                        loss = compute_task_loss_with_focal(
                            outputs['logits'],
                            labels,
                            task_type=self.config.task,
                            ignore_index=-100,
                            use_focal=self.use_focal_loss,
                            focal_gamma=self.focal_gamma,
                            label_smoothing=self.label_smoothing,
                        )
                        outputs['loss'] = loss
                        outputs['task_loss'] = loss
                    else:
                        loss = None
                else:
                    loss = outputs.get('loss', outputs.get('total_loss'))

                if loss is None:
                    # Skip batch if loss couldn't be computed
                    continue
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate losses
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            for key, value in outputs.items():
                if 'loss' in key and torch.is_tensor(value):
                    loss_components[key] = loss_components.get(key, 0) + value.item()

            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                # Use normal scheduler unless SWA is active
                swa_start = getattr(self.config, 'swa_start_epoch', 50)
                if self.swa_model is not None and self.current_epoch >= swa_start:
                    self.swa_scheduler.step()
                else:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Update EMA weights after each optimizer step
                if self.ema is not None:
                    self.ema.update()

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())

                # Log to WandB periodically
                if self.config.use_wandb and self.global_step % self.config.log_every_n_steps == 0:
                    _get_wandb().log({
                        'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                        'train/lr': self.scheduler.get_last_lr()[0],
                    }, step=self.global_step)

        # Compute epoch metrics
        n_steps = len(self.train_dataloader)
        metrics = {
            'train_loss': total_loss / n_steps,
            **{f'train_{k}': v / n_steps for k, v in loss_components.items()}
        }

        return metrics

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation on the eval dataset."""
        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0.0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            outputs = self.model(
                images=batch.get('pixel_values'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                bbox=batch.get('bbox'),
                labels=batch.get('labels'),
                start_positions=batch.get('start_positions'),
                end_positions=batch.get('end_positions'),
                return_loss=True,
            )

            loss_val = outputs.get('loss', outputs.get('total_loss'))
            if loss_val is not None and torch.is_tensor(loss_val):
                total_loss += loss_val.item()

            # Get predictions and labels based on task type
            if self.config.task == 'vqa':
                # VQA: predictions are start/end logits
                if 'start_logits' in outputs:
                    start_preds = outputs['start_logits'].argmax(dim=-1)
                    end_preds = outputs['end_logits'].argmax(dim=-1)
                    all_predictions.extend(list(zip(
                        start_preds.cpu().numpy().tolist(),
                        end_preds.cpu().numpy().tolist()
                    )))
                    if 'start_positions' in batch:
                        all_labels.extend(list(zip(
                            batch['start_positions'].cpu().numpy().tolist(),
                            batch['end_positions'].cpu().numpy().tolist()
                        )))
                else:
                    # Fallback for models without separate start/end logits
                    logits = outputs.get('logits', outputs.get('start_logits'))
                    if logits is not None:
                        predictions = logits.argmax(dim=-1)
                        all_predictions.extend(predictions.cpu().numpy().tolist())
            else:
                # NER/Classification: standard logits
                logits = outputs['logits']
                predictions = logits.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().numpy().tolist())
                if 'labels' in batch:
                    all_labels.extend(batch['labels'].cpu().numpy().tolist())

        # Compute metrics based on task type
        from ..evaluation.metrics import compute_metrics, compute_ner_f1

        if self.config.task == 'ner' and self.config.id2label is not None:
            # Compute proper NER F1 using seqeval
            ner_metrics = compute_ner_f1(
                all_predictions,
                all_labels,
                self.config.id2label,
                ignore_index=-100
            )
            metrics = {
                'eval_f1': ner_metrics['f1'],
                'eval_precision': ner_metrics['precision'],
                'eval_recall': ner_metrics['recall'],
            }
        elif self.config.task == 'vqa':
            # VQA: compute exact match for start/end positions
            if all_labels and all_predictions:
                exact_match = sum(
                    1 for pred, label in zip(all_predictions, all_labels)
                    if pred == label
                ) / len(all_labels)
                metrics = {
                    'eval_exact_match': exact_match,
                    'eval_f1': exact_match,  # Use EM as proxy for F1 in VQA
                }
            else:
                metrics = {'eval_f1': 0.0, 'eval_exact_match': 0.0}
        else:
            metrics = compute_metrics(
                all_predictions,
                all_labels,
                task=self.config.task,
                id2label=self.config.id2label
            )

        metrics['eval_loss'] = total_loss / max(len(self.eval_dataloader), 1)

        return metrics

    def _get_curriculum_severity(self, epoch: int) -> int:
        """Get curriculum severity level based on epoch."""
        if epoch < self.config.curriculum_warmup_epochs:
            return self.config.curriculum_start_severity

        progress = (epoch - self.config.curriculum_warmup_epochs) / \
                  (self.config.epochs - self.config.curriculum_warmup_epochs)

        severity_range = self.config.curriculum_end_severity - \
                        self.config.curriculum_start_severity

        return int(self.config.curriculum_start_severity + progress * severity_range)

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save training state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
        }, checkpoint_dir / 'training_state.pt')

        # Save config
        with open(checkpoint_dir / 'training_config.json', 'w') as f:
            json.dump(vars(self.config), f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_path)

        # Load model
        self.model = CaliDocModel.from_pretrained(str(checkpoint_dir))
        self.model.to(self.device)

        # Load training state
        state_path = checkpoint_dir / 'training_state.pt'
        if state_path.exists():
            state = torch.load(state_path)
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.global_step = state['global_step']
            self.current_epoch = state['epoch']
            self.best_metric = state['best_metric']
            # train() continues from the epoch after the checkpointed one.
            self._resume_epoch = state['epoch'] + 1

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
