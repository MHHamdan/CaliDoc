#!/usr/bin/env python3
"""
CaliDoc Training Script (Phase 2)

All data comes from the verified bundle; the label schema is loaded from the
bundle's label_list.json (never hardcoded). Early stopping reads ONLY val_es
(CORD) / es_holdout (FUNSD). Conformal calibration happens in a separate
evaluation stage — never inside training.

Usage:
    python scripts/train.py experiment=cord_phase2 experiment.seed=42
    python scripts/train.py experiment=funsd_phase2 paths.bundle_dir=data_bundle
"""

import json
import os
import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calidoc.models import CaliDocModel
from calidoc.data import create_dataloader, verify_bundle
from calidoc.data.funsd_dataset import FUNSDDataset
from calidoc.data.cord_dataset import CORDDataset
from calidoc.data.sroie_dataset import SROIEDataset
from calidoc.data.wildreceipt_dataset import WildReceiptDataset, WildReceiptTraincalDataset
from calidoc.training import CaliDocTrainer
from calidoc.training.trainer import TrainingConfig
from calidoc.utils import set_seed, setup_logging, save_experiment_config


logger = logging.getLogger(__name__)

# The one split allowed to drive early stopping / temperature, per dataset.
ES_SPLIT = {'cord': 'val_es', 'funsd': 'es_holdout', 'sroie': 'val_es', 'wildreceipt': 'val_es',
            'wildreceipt_traincal': 'val_es'}


def create_dataset(cfg: DictConfig, split: str):
    """Create a fit-stage dataset from the verified bundle."""
    dataset_name = cfg.dataset.name
    bundle_dir = Path(cfg.paths.bundle_dir)

    if dataset_name == 'funsd':
        cls = FUNSDDataset
    elif dataset_name == 'cord':
        cls = CORDDataset
    elif dataset_name == 'sroie':
        cls = SROIEDataset
    elif dataset_name == 'wildreceipt_traincal':
        cls = WildReceiptTraincalDataset
    elif dataset_name == 'wildreceipt':
        cls = WildReceiptDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return cls(
        bundle_dir=str(bundle_dir),
        split=split,
        max_length=cfg.dataset.processing.max_length,
        apply_augmentation=cfg.training.robustness.augmentation.enabled if split == 'train' else False,
        stage='fit',
    )


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    # Setup
    setup_logging(level=logging.INFO)
    set_seed(cfg.experiment.seed)

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_experiment_config(
        OmegaConf.to_container(cfg, resolve=True),
        str(output_dir)
    )

    # Startup protocol assertions: label count, zero-O CORD gold, FUNSD O
    # fraction, split disjointness. Crashes before any GPU time is spent.
    bundle_manifest = verify_bundle(cfg.paths.bundle_dir, cfg.dataset.name)
    if cfg.dataset.name == 'cord' and bundle_manifest['num_labels'] != 54:
        raise AssertionError(f"CORD must have 54 labels, got {bundle_manifest['num_labels']}")

    run_manifest = {
        **bundle_manifest,
        'experiment': cfg.experiment.name,
        'seed': cfg.experiment.seed,
        'es_split': ES_SPLIT[cfg.dataset.name],
        'legacy_invalid_note': (
            'Phase 1 CORD numbers are legacy_invalid (label-schema bug); '
            'never compare against them.'
        ),
    }
    with open(output_dir / 'run_manifest.json', 'w') as f:
        json.dump(run_manifest, f, indent=2, default=str)
    logger.info(f"Run manifest written; B-only fields: {bundle_manifest['b_only_fields']}")

    # Create datasets. Early stopping + temperature fitting read ONLY the ES
    # split; conformal calibration docs are unreachable at stage='fit' (the
    # dataset constructor crashes on any leakage).
    logger.info("Creating datasets...")
    train_dataset = create_dataset(cfg, 'train')
    eval_dataset = create_dataset(cfg, ES_SPLIT[cfg.dataset.name])

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.hardware.num_workers,
    )

    eval_dataloader = create_dataloader(
        eval_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.hardware.num_workers,
    )

    # Create model
    logger.info(f"Creating model: {cfg.model.name}")

    # Get num_labels from dataset (for NER tasks)
    num_labels = getattr(train_dataset, 'num_labels', None)
    if num_labels is None:
        # Try to get from config (OmegaConf)
        num_labels = cfg.dataset.get('num_labels', None)

    task_type = 'vqa' if cfg.dataset.task == 'vqa' else 'ner'
    logger.info(f"Task type: {task_type}, num_labels: {num_labels}")

    # Validate num_labels for NER tasks
    if task_type == 'ner' and num_labels is None:
        raise ValueError("num_labels must be specified for NER tasks in dataset config")

    encoder_config = {
        'model_name': cfg.model.pretrained_name,
        'task': task_type,
        'num_labels': num_labels,
    }

    calibration_config = {
        'alpha': cfg.calibration.alpha,
        'score_type': cfg.calibration.score_type,
        'lambda_layout': cfg.calibration.lambda_layout,
        'lambda_ocr': cfg.calibration.lambda_ocr,
    }

    model = CaliDocModel(
        encoder_type=cfg.model.name,
        encoder_config=encoder_config,
        calibration_config=calibration_config,
        use_quality_routing=cfg.routing.enabled,
        use_calibration=cfg.calibration.enabled,
    )

    # Get id2label mapping from dataset for NER evaluation
    id2label = getattr(train_dataset, 'id2label', None)

    # Create training config
    # Get optional SOTA settings
    llrd_rate = cfg.training.get('llrd_rate', cfg.training.optimizer.get('llrd_rate', 1.0))
    use_ema = cfg.training.get('use_ema', False)
    ema_decay = cfg.training.get('ema_decay', 0.999)
    label_smoothing = cfg.training.get('label_smoothing', 0.0)
    use_focal_loss = cfg.training.get('use_focal_loss', False)
    focal_gamma = cfg.training.get('focal_gamma', 2.0)

    training_config = TrainingConfig(
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.optimizer.learning_rate,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_ratio=cfg.training.scheduler.warmup_ratio,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        task_weight=cfg.training.losses.task_weight,
        calibration_weight=cfg.training.losses.calibration_weight,
        quality_weight=cfg.training.losses.quality_weight,
        routing_weight=cfg.training.losses.routing_weight,
        output_dir=str(output_dir),
        mixed_precision=cfg.hardware.mixed_precision,
        device=cfg.hardware.device,
        use_wandb=cfg.logging.use_wandb,
        wandb_project=cfg.get('wandb', {}).get('project', 'calidoc'),
        task=task_type,
        id2label=id2label,
        llrd_rate=llrd_rate,
        use_ema=use_ema,
        ema_decay=ema_decay,
        early_stopping_patience=(
            cfg.training.early_stopping.patience
            if cfg.training.get('early_stopping', {}).get('enabled', False) else 0
        ),
    )

    # Store additional config for loss function
    training_config.label_smoothing = label_smoothing
    training_config.use_focal_loss = use_focal_loss
    training_config.focal_gamma = focal_gamma

    # Create trainer
    trainer = CaliDocTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
    )

    # Resume if a checkpoint is available (SLURM requeue support)
    resume_from = cfg.training.get('resume_from', None)
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = output_dir / resume_path
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            trainer.load_checkpoint(str(resume_path))
        else:
            logger.info(f"No checkpoint at {resume_path}, starting fresh")

    # Train
    logger.info("Starting training...")
    metrics = trainer.train()

    logger.info(f"Training complete. Best metrics: {metrics}")

    # NOTE: conformal calibration is deliberately NOT performed here. It runs
    # in the separate conformal evaluation stage against the persisted cal
    # splits (Phase 2 protocol); carving calibration data out of the ES split
    # inside training was a protocol violation.

    # Save final model
    model.save_pretrained(str(output_dir / 'final'))
    logger.info(f"Model saved to {output_dir / 'final'}")

    return metrics


if __name__ == "__main__":
    main()
