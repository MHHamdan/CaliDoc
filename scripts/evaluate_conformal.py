#!/usr/bin/env python3
"""
CaliDoc Conformal Prediction Evaluation Script

This script runs comprehensive experiments for the CVPR paper:
1. Baseline accuracy evaluation
2. Conformal prediction with coverage guarantees
3. Selective prediction (risk-coverage curves)
4. Calibration analysis (ECE, reliability diagrams)
5. Robustness under perturbations

Usage:
    python scripts/evaluate_conformal.py \
        --checkpoint_path outputs/funsd_aug/best \
        --dataset funsd \
        --output_dir results/conformal

    # For robustness evaluation
    python scripts/evaluate_conformal.py \
        --checkpoint_path outputs/funsd_aug/best \
        --dataset funsd \
        --output_dir results/robustness \
        --run_robustness
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calidoc.evaluation.conformal import (
    ConformalPredictor,
    SelectivePredictor,
    compute_calibration_error,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_data(
    checkpoint_path: str,
    dataset_name: str,
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pretrained_name: str = 'microsoft/layoutlmv3-large',
):
    """Load model checkpoint and dataset."""
    from transformers import AutoProcessor
    from calidoc.models import CaliDocModel

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Determine task and num_labels based on dataset
    # FUNSD and CORD are NER (token classification) tasks
    num_labels = None
    if dataset_name in ['funsd', 'cord']:
        task = 'ner'
        # Try to read num_labels from checkpoint's training_config.json
        training_config_path = os.path.join(checkpoint_path, 'training_config.json')
        if os.path.exists(training_config_path):
            import json
            with open(training_config_path) as f:
                training_config = json.load(f)
            num_labels = training_config.get('num_labels', None)
            if num_labels:
                logger.info(f"Read num_labels={num_labels} from training config")

        # Fallback to defaults if not found
        if num_labels is None:
            # FUNSD: 7 labels, CORD: 53 labels (with BIO tagging)
            num_labels = 7 if dataset_name == 'funsd' else 53
            logger.info(f"Using default num_labels={num_labels} for {dataset_name}")
    else:
        task = 'vqa'
        num_labels = None

    # Load CaliDocModel from checkpoint with correct task type
    encoder_config = {
        'model_name': pretrained_name,
        'task': task,
        'num_labels': num_labels,
    }
    # Disable quality routing and calibration for evaluation
    # (these may not have been trained or saved)
    model = CaliDocModel.from_pretrained(
        checkpoint_path,
        encoder_config=encoder_config,
        use_quality_routing=False,
        use_calibration=False,
    )

    # Load processor from HuggingFace (not from checkpoint)
    logger.info(f"Loading processor from {pretrained_name}")
    processor = AutoProcessor.from_pretrained(pretrained_name, apply_ocr=False)

    # Load dataset
    logger.info(f"Loading {dataset_name} dataset from {data_root}")

    # Get dataset with splits
    if dataset_name == 'funsd':
        from calidoc.data.funsd_dataset import FUNSDDataset
        val_dataset = FUNSDDataset(
            data_dir=os.path.join(data_root, 'funsd'),
            processor=processor,
            split='test',  # FUNSD uses test split for evaluation
            max_length=512,
        )
    elif dataset_name == 'cord':
        from calidoc.data.cord_dataset import CORDDataset
        val_dataset = CORDDataset(
            data_dir=os.path.join(data_root, 'cord'),
            processor=processor,
            split='test',
            max_length=512,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None,
    )

    # Get label mapping from dataset
    id2label = val_dataset.id2label if hasattr(val_dataset, 'id2label') else None
    num_labels = val_dataset.num_labels if hasattr(val_dataset, 'num_labels') else None

    return model, processor, val_loader, id2label, num_labels


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Collect all predictions and labels from the dataset.

    Returns:
        Dictionary with 'logits', 'labels', 'mask' tensors
    """
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Forward pass (CaliDocModel interface)
            outputs = model(
                images=batch.get('pixel_values'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                bbox=batch.get('bbox'),
                labels=batch.get('labels'),
                return_loss=False,
            )

            # CaliDocModel returns dict with 'logits' key
            logits = outputs['logits']
            labels = batch['labels']
            attention_mask = batch['attention_mask']

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(attention_mask.cpu())

    return {
        'logits': torch.cat(all_logits, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'mask': torch.cat(all_masks, dim=0),
    }


def run_baseline_evaluation(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    id2label: Optional[Dict] = None,
) -> Dict[str, float]:
    """Run standard accuracy/F1 evaluation."""
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

    # Get predictions
    predictions = logits.argmax(dim=-1)

    # Convert to label sequences (filtering padding and ignored tokens)
    pred_labels = []
    true_labels = []

    for i in range(len(predictions)):
        pred_seq = []
        true_seq = []

        for j in range(len(predictions[i])):
            if mask[i][j] == 0:
                continue
            if labels[i][j] < 0:
                continue

            pred_label = predictions[i][j].item()
            true_label = labels[i][j].item()

            if id2label:
                pred_seq.append(id2label.get(pred_label, f"LABEL_{pred_label}"))
                true_seq.append(id2label.get(true_label, f"LABEL_{true_label}"))
            else:
                pred_seq.append(f"LABEL_{pred_label}")
                true_seq.append(f"LABEL_{true_label}")

        if pred_seq:
            pred_labels.append(pred_seq)
            true_labels.append(true_seq)

    # Compute metrics
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    logger.info(f"Baseline F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': classification_report(true_labels, pred_labels),
    }


def run_conformal_evaluation(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    target_coverages: List[float] = [0.90, 0.95, 0.99],
    calibration_ratio: float = 0.5,
) -> Dict[str, any]:
    """
    Run conformal prediction evaluation.

    Args:
        logits: Model logits [batch, seq_len, num_classes]
        labels: True labels [batch, seq_len]
        mask: Attention mask [batch, seq_len]
        target_coverages: List of target coverage rates
        calibration_ratio: Fraction of data for calibration

    Returns:
        Dictionary with conformal prediction results
    """
    num_classes = logits.size(-1)

    # Flatten data
    flat_logits = logits.view(-1, num_classes)
    flat_labels = labels.view(-1)
    flat_mask = mask.view(-1)

    # Filter valid tokens
    valid = flat_mask.bool() & (flat_labels >= 0)
    valid_logits = flat_logits[valid]
    valid_labels = flat_labels[valid]

    # Split into calibration and test sets
    n_samples = len(valid_logits)
    n_cal = int(n_samples * calibration_ratio)

    # Shuffle indices
    perm = torch.randperm(n_samples)
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    cal_logits = valid_logits[cal_idx]
    cal_labels = valid_labels[cal_idx]
    test_logits = valid_logits[test_idx]
    test_labels = valid_labels[test_idx]

    logger.info(f"Calibration set: {len(cal_logits)}, Test set: {len(test_logits)}")

    results = {}

    for target_cov in target_coverages:
        logger.info(f"\n{'='*50}")
        logger.info(f"Target coverage: {target_cov*100:.0f}%")

        # LAC method
        lac_predictor = ConformalPredictor(
            num_classes=num_classes,
            target_coverage=target_cov,
            method='lac',
        )
        lac_predictor.calibrate(cal_logits, cal_labels)
        lac_metrics = lac_predictor.evaluate(test_logits, test_labels)

        logger.info(f"LAC - Coverage: {lac_metrics['coverage']*100:.2f}%, "
                   f"Avg Set Size: {lac_metrics['avg_set_size']:.2f}")

        # APS method
        aps_predictor = ConformalPredictor(
            num_classes=num_classes,
            target_coverage=target_cov,
            method='aps',
        )
        aps_predictor.calibrate(cal_logits, cal_labels)
        aps_metrics = aps_predictor.evaluate(test_logits, test_labels)

        logger.info(f"APS - Coverage: {aps_metrics['coverage']*100:.2f}%, "
                   f"Avg Set Size: {aps_metrics['avg_set_size']:.2f}")

        results[f'coverage_{int(target_cov*100)}'] = {
            'target': target_cov,
            'lac': lac_metrics,
            'aps': aps_metrics,
        }

    return results


def run_selective_prediction(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    id2label: Optional[Dict] = None,
) -> Dict[str, any]:
    """
    Run selective prediction evaluation (risk-coverage curves).
    """
    from seqeval.metrics import f1_score

    num_classes = logits.size(-1)

    # Get predictions and confidence
    probs = F.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)

    # Coverage levels to evaluate
    coverage_levels = np.linspace(0.5, 1.0, 11)

    results = {
        'coverage': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'threshold': [],
    }

    for target_cov in coverage_levels:
        # Flatten and filter valid tokens
        flat_conf = confidence.view(-1)
        flat_pred = predictions.view(-1)
        flat_labels = labels.view(-1)
        flat_mask = mask.view(-1)

        valid = flat_mask.bool() & (flat_labels >= 0)
        valid_conf = flat_conf[valid]
        valid_pred = flat_pred[valid]
        valid_labels = flat_labels[valid]

        # Find threshold for target coverage
        n = len(valid_conf)
        k = int(np.ceil(n * target_cov))
        sorted_conf, _ = torch.sort(valid_conf, descending=True)
        threshold = sorted_conf[k - 1].item() if k > 0 else 0

        # Accept high-confidence predictions
        accepted = valid_conf >= threshold

        # Compute F1 on accepted predictions
        accepted_pred = valid_pred[accepted]
        accepted_labels = valid_labels[accepted]

        # Convert to sequences for seqeval
        pred_seq = [id2label.get(p.item(), f"L{p.item()}") for p in accepted_pred] if id2label else [f"L{p.item()}" for p in accepted_pred]
        true_seq = [id2label.get(l.item(), f"L{l.item()}") for l in accepted_labels] if id2label else [f"L{l.item()}" for l in accepted_labels]

        # Wrap in list for seqeval
        if pred_seq:
            f1 = f1_score([true_seq], [pred_seq])
        else:
            f1 = 0

        actual_coverage = accepted.sum().item() / n

        results['coverage'].append(actual_coverage)
        results['f1'].append(f1)
        results['threshold'].append(threshold)

        logger.info(f"Coverage {target_cov*100:.0f}%: F1={f1*100:.2f}%, Threshold={threshold:.3f}")

    return results


def run_calibration_analysis(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, any]:
    """Run calibration analysis (ECE, reliability diagrams)."""
    logger.info("\nRunning calibration analysis...")

    cal_results = compute_calibration_error(logits, labels, n_bins=15, mask=mask)

    logger.info(f"ECE: {cal_results['ece']*100:.2f}%")
    logger.info(f"MCE: {cal_results['mce']*100:.2f}%")

    return cal_results


def apply_perturbation_to_batch(
    batch: Dict[str, torch.Tensor],
    perturbation_fn: callable,
    severity: int,
) -> Dict[str, torch.Tensor]:
    """Apply perturbation to pixel_values in batch."""
    import cv2

    perturbed_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

    if 'pixel_values' in perturbed_batch:
        pixel_values = perturbed_batch['pixel_values']
        batch_size = pixel_values.shape[0]

        perturbed_pixels = []
        for i in range(batch_size):
            # Convert tensor to numpy image (C, H, W) -> (H, W, C)
            img = pixel_values[i].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))

            # Denormalize (LayoutLMv3 uses ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = (img * 255).astype(np.uint8)

            # Apply perturbation
            try:
                perturbed = perturbation_fn(img, severity=severity)
            except Exception as e:
                logger.warning(f"Perturbation failed: {e}, using original")
                perturbed = img

            # Renormalize
            perturbed = perturbed.astype(np.float32) / 255.0
            perturbed = (perturbed - mean) / std
            perturbed = np.transpose(perturbed, (2, 0, 1))
            perturbed_pixels.append(torch.from_numpy(perturbed).float())

        perturbed_batch['pixel_values'] = torch.stack(perturbed_pixels)

    return perturbed_batch


def collect_predictions_with_perturbation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    perturbation_fn: callable,
    severity: int,
) -> Dict[str, torch.Tensor]:
    """Collect predictions with perturbations applied to images."""
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            # Apply perturbation to batch
            perturbed_batch = apply_perturbation_to_batch(batch, perturbation_fn, severity)

            # Move to device
            perturbed_batch = {k: v.to(device) if torch.is_tensor(v) else v
                              for k, v in perturbed_batch.items()}

            # Forward pass
            outputs = model(
                images=perturbed_batch.get('pixel_values'),
                input_ids=perturbed_batch.get('input_ids'),
                attention_mask=perturbed_batch.get('attention_mask'),
                bbox=perturbed_batch.get('bbox'),
                labels=perturbed_batch.get('labels'),
                return_loss=False,
            )

            logits = outputs['logits']
            labels = perturbed_batch['labels']
            attention_mask = perturbed_batch['attention_mask']

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(attention_mask.cpu())

    return {
        'logits': torch.cat(all_logits, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'mask': torch.cat(all_masks, dim=0),
    }


def run_robustness_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Optional[Dict] = None,
    severities: List[int] = [1, 2, 3],
) -> Dict[str, any]:
    """
    Run robustness evaluation under perturbations.
    """
    from calidoc.data.perturbations import DocumentPerturbations
    from seqeval.metrics import f1_score

    logger.info("\nRunning robustness evaluation...")

    # Get all perturbations
    all_perturbations = DocumentPerturbations.get_all_perturbations()

    # Select key perturbations for evaluation
    perturbation_names = [
        'gaussian_blur', 'motion_blur',
        'gaussian_noise', 'salt_pepper_noise',
        'jpeg_compression',
        'brightness', 'contrast',
        'ink_bleed', 'paper_texture',
    ]

    results = {'clean': {}, 'perturbed': {}, 'summary': {}}

    # First, evaluate on clean data
    logger.info("Evaluating on clean data...")
    clean_preds = collect_predictions(model, dataloader, device)
    clean_metrics = run_baseline_evaluation(
        clean_preds['logits'],
        clean_preds['labels'],
        clean_preds['mask'],
        id2label,
    )
    results['clean'] = {'f1': clean_metrics['f1']}
    clean_f1 = clean_metrics['f1']

    logger.info(f"Clean F1: {clean_f1*100:.2f}%")

    # Evaluate under each perturbation
    for perturb_name in perturbation_names:
        if perturb_name not in all_perturbations:
            logger.warning(f"Perturbation {perturb_name} not found, skipping")
            continue

        perturbation_fn = all_perturbations[perturb_name]
        results['perturbed'][perturb_name] = {}

        for severity in severities:
            logger.info(f"Evaluating {perturb_name} (severity {severity})...")

            try:
                # Collect predictions with perturbation applied
                perturbed_preds = collect_predictions_with_perturbation(
                    model, dataloader, device, perturbation_fn, severity
                )

                # Evaluate
                perturbed_metrics = run_baseline_evaluation(
                    perturbed_preds['logits'],
                    perturbed_preds['labels'],
                    perturbed_preds['mask'],
                    id2label,
                )

                perturbed_f1 = perturbed_metrics['f1']
                f1_drop = clean_f1 - perturbed_f1
                relative_drop = f1_drop / clean_f1 if clean_f1 > 0 else 0

                results['perturbed'][perturb_name][f'severity_{severity}'] = {
                    'f1': perturbed_f1,
                    'f1_drop': f1_drop,
                    'relative_drop': relative_drop,
                }

                logger.info(f"  F1: {perturbed_f1*100:.2f}% (drop: {f1_drop*100:.2f}%)")

            except Exception as e:
                logger.error(f"  Error evaluating {perturb_name} severity {severity}: {e}")
                results['perturbed'][perturb_name][f'severity_{severity}'] = {
                    'f1': 0,
                    'f1_drop': 0,
                    'error': str(e),
                }

    # Compute summary statistics
    all_drops = []
    for perturb_name, sev_results in results['perturbed'].items():
        for sev_key, metrics in sev_results.items():
            if 'relative_drop' in metrics:
                all_drops.append(metrics['relative_drop'])

    if all_drops:
        results['summary'] = {
            'mean_relative_drop': np.mean(all_drops),
            'max_relative_drop': np.max(all_drops),
            'min_relative_drop': np.min(all_drops),
            'num_perturbations': len(perturbation_names),
            'num_severities': len(severities),
        }
        logger.info(f"\nSummary: Mean relative F1 drop: {results['summary']['mean_relative_drop']*100:.2f}%")

    return results


def save_results(results: Dict, output_dir: str, prefix: str = ""):
    """Save results to JSON and generate summary."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}results_{timestamp}.json" if prefix else f"results_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    results_json = convert_for_json(results)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    return output_path


def generate_latex_tables(results: Dict, output_dir: str):
    """Generate LaTeX tables for the paper."""
    os.makedirs(output_dir, exist_ok=True)

    # Conformal prediction table
    if 'conformal' in results:
        latex = """
\\begin{table}[t]
\\centering
\\caption{Conformal Prediction Results on FUNSD}
\\label{tab:conformal}
\\begin{tabular}{lccc}
\\toprule
Target Coverage & Method & Empirical Coverage & Avg Set Size \\\\
\\midrule
"""
        for cov_key, cov_data in results['conformal'].items():
            target = cov_data['target']
            lac = cov_data['lac']
            aps = cov_data['aps']

            latex += f"{target*100:.0f}\\% & LAC & {lac['coverage']*100:.1f}\\% & {lac['avg_set_size']:.2f} \\\\\n"
            latex += f" & APS & {aps['coverage']*100:.1f}\\% & {aps['avg_set_size']:.2f} \\\\\n"
            latex += "\\midrule\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        with open(os.path.join(output_dir, 'conformal_table.tex'), 'w') as f:
            f.write(latex)

    # Selective prediction table
    if 'selective' in results:
        latex = """
\\begin{table}[t]
\\centering
\\caption{Selective Prediction Results}
\\label{tab:selective}
\\begin{tabular}{lcc}
\\toprule
Coverage & F1 Score & Improvement \\\\
\\midrule
"""
        baseline_f1 = results.get('baseline', {}).get('f1', 0)

        for cov, f1 in zip(results['selective']['coverage'], results['selective']['f1']):
            improvement = (f1 - baseline_f1) * 100 if baseline_f1 > 0 else 0
            latex += f"{cov*100:.0f}\\% & {f1*100:.1f}\\% & +{improvement:.1f}\\% \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        with open(os.path.join(output_dir, 'selective_table.tex'), 'w') as f:
            f.write(latex)

    logger.info(f"LaTeX tables saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CaliDoc Conformal Prediction Evaluation")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='funsd',
                        choices=['funsd', 'cord'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./results/conformal',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--run_robustness', action='store_true',
                        help='Run robustness evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--pretrained_name', type=str, default='microsoft/layoutlmv3-large',
                        help='HuggingFace model name for processor')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model and data
    model, processor, val_loader, id2label, num_labels = load_model_and_data(
        args.checkpoint_path,
        args.dataset,
        args.data_root,
        args.batch_size,
        args.num_workers,
        args.pretrained_name,
    )

    # Collect predictions
    logger.info("\nCollecting predictions...")
    predictions = collect_predictions(model, val_loader, device)

    results = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint_path,
        'timestamp': datetime.now().isoformat(),
    }

    # 1. Baseline evaluation
    logger.info("\n" + "="*60)
    logger.info("BASELINE EVALUATION")
    logger.info("="*60)
    results['baseline'] = run_baseline_evaluation(
        predictions['logits'],
        predictions['labels'],
        predictions['mask'],
        id2label,
    )

    # 2. Conformal prediction
    logger.info("\n" + "="*60)
    logger.info("CONFORMAL PREDICTION EVALUATION")
    logger.info("="*60)
    results['conformal'] = run_conformal_evaluation(
        predictions['logits'],
        predictions['labels'],
        predictions['mask'],
    )

    # 3. Selective prediction
    logger.info("\n" + "="*60)
    logger.info("SELECTIVE PREDICTION EVALUATION")
    logger.info("="*60)
    results['selective'] = run_selective_prediction(
        predictions['logits'],
        predictions['labels'],
        predictions['mask'],
        id2label,
    )

    # 4. Calibration analysis
    logger.info("\n" + "="*60)
    logger.info("CALIBRATION ANALYSIS")
    logger.info("="*60)
    results['calibration'] = run_calibration_analysis(
        predictions['logits'],
        predictions['labels'],
        predictions['mask'],
    )

    # 5. Robustness evaluation (optional)
    if args.run_robustness:
        logger.info("\n" + "="*60)
        logger.info("ROBUSTNESS EVALUATION")
        logger.info("="*60)
        results['robustness'] = run_robustness_evaluation(
            model, val_loader, device, id2label,
        )

    # Save results
    save_results(results, args.output_dir)
    generate_latex_tables(results, args.output_dir)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Baseline F1: {results['baseline']['f1']*100:.2f}%")
    logger.info(f"ECE: {results['calibration']['ece']*100:.2f}%")

    if 'conformal' in results:
        for cov_key, cov_data in results['conformal'].items():
            logger.info(f"Conformal {cov_data['target']*100:.0f}%: "
                       f"Coverage={cov_data['lac']['coverage']*100:.1f}%, "
                       f"Set Size={cov_data['lac']['avg_set_size']:.2f}")

    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
