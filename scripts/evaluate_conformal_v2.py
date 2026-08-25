#!/usr/bin/env python3
"""
CaliDoc Conformal Prediction Evaluation v2 — Structure-Aware Scoring

This script uses the structure-aware ConformalCalibrator from
calidoc.models.conformal_calibrator (with layout entropy + OCR confidence
terms) instead of the simpler softmax-only evaluation/conformal.py.

Key differences from v1 (evaluate_conformal.py):
  1. Imports and uses ConformalCalibrator with structure-aware scoring
  2. Collects layout_attention from model outputs during inference
  3. Uses proper data splits: CORD val→calibrate, test→eval;
     FUNSD test split 70/30 (no separate val set)
  4. Runs ablation across LAC/APS × softmax/structure-aware
  5. Includes temperature scaling baseline
  6. Saves per-token predictions to JSON for qualitative visualization
  7. Computes per-class coverage and set sizes

Usage:
    python scripts/evaluate_conformal_v2.py \
        --checkpoint_path outputs/cord_optimized_6530869/best \
        --dataset cord \
        --data_root data \
        --output_dir results/conformal_v2_cord \
        --save_per_token \
        --use_structure_aware
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calidoc.models.conformal_calibrator import ConformalCalibrator, CalibrationResult

# Also import the simple predictor for baseline comparison
from calidoc.evaluation.conformal import (
    ConformalPredictor,
    compute_calibration_error,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_and_data(
    checkpoint_path: str,
    dataset_name: str,
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pretrained_name: str = 'microsoft/layoutlmv3-large',
):
    """Load model checkpoint and dataset(s).

    Returns model, processor, dataloaders dict, id2label, num_labels.
    For CORD the dict has 'val' and 'test'; for FUNSD only 'test'.
    """
    from transformers import AutoProcessor
    from calidoc.models import CaliDocModel

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Determine task and num_labels
    num_labels = None
    if dataset_name in ['funsd', 'cord']:
        task = 'ner'
        training_config_path = os.path.join(checkpoint_path, 'training_config.json')
        if os.path.exists(training_config_path):
            with open(training_config_path) as f:
                training_config = json.load(f)
            num_labels = training_config.get('num_labels', None)
        if num_labels is None:
            num_labels = 7 if dataset_name == 'funsd' else 53
            logger.info(f"Using default num_labels={num_labels} for {dataset_name}")
    else:
        task = 'vqa'
        num_labels = None

    encoder_config = {
        'model_name': pretrained_name,
        'task': task,
        'num_labels': num_labels,
    }
    model = CaliDocModel.from_pretrained(
        checkpoint_path,
        encoder_config=encoder_config,
        use_quality_routing=False,
        use_calibration=False,
    )

    processor = AutoProcessor.from_pretrained(pretrained_name, apply_ocr=False)

    loaders: Dict[str, DataLoader] = {}
    id2label = None

    # Determine whether to use local Arrow data or HF cache
    def _has_local_arrow(dname, split):
        p = os.path.join(data_root, dname, split)
        return os.path.isdir(p) and os.path.isfile(os.path.join(p, 'dataset_info.json'))

    if dataset_name == 'funsd':
        from calidoc.data.funsd_dataset import FUNSDDataset
        use_local = _has_local_arrow('funsd', 'test')
        logger.info(f"FUNSD test: local Arrow={'found' if use_local else 'NOT found'}")
        test_dataset = FUNSDDataset(
            data_dir=os.path.join(data_root, 'funsd'),
            processor=processor,
            split='test',
            max_length=512,
            use_huggingface=not use_local,
        )
        logger.info(f"FUNSD test: loaded {len(test_dataset)} samples")
        loaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=getattr(test_dataset, 'collate_fn', None),
        )
        id2label = getattr(test_dataset, 'id2label', None)
        num_labels = getattr(test_dataset, 'num_labels', num_labels)

    elif dataset_name == 'cord':
        from calidoc.data.cord_dataset import CORDDataset
        # CORD has a dedicated validation split for calibration
        use_local_val = _has_local_arrow('cord', 'validation')
        use_local_test = _has_local_arrow('cord', 'test')
        logger.info(f"CORD val: local Arrow={'found' if use_local_val else 'NOT found'}")
        logger.info(f"CORD test: local Arrow={'found' if use_local_test else 'NOT found'}")
        val_dataset = CORDDataset(
            data_dir=os.path.join(data_root, 'cord'),
            processor=processor,
            split='validation',
            max_length=512,
            use_huggingface=not use_local_val,
        )
        test_dataset = CORDDataset(
            data_dir=os.path.join(data_root, 'cord'),
            processor=processor,
            split='test',
            max_length=512,
            use_huggingface=not use_local_test,
        )
        logger.info(f"CORD val: {len(val_dataset)} samples, test: {len(test_dataset)} samples")
        loaders['val'] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=getattr(val_dataset, 'collate_fn', None),
        )
        loaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            collate_fn=getattr(test_dataset, 'collate_fn', None),
        )
        id2label = getattr(test_dataset, 'id2label', None)
        num_labels = getattr(test_dataset, 'num_labels', num_labels)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return model, processor, loaders, id2label, num_labels


# ---------------------------------------------------------------------------
# Inference — collects layout_attention alongside logits
# ---------------------------------------------------------------------------

def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Collect predictions, labels, masks, *and* layout_attention."""
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []
    all_masks = []
    all_layout_attention = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            outputs = model(
                images=batch.get('pixel_values'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                bbox=batch.get('bbox'),
                labels=batch.get('labels'),
                return_loss=False,
            )

            logits = outputs['logits']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            layout_attn = outputs.get('layout_attention')

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(attention_mask.cpu())
            if layout_attn is not None:
                all_layout_attention.append(layout_attn.cpu())

    result = {
        'logits': torch.cat(all_logits, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'mask': torch.cat(all_masks, dim=0),
    }
    if all_layout_attention:
        result['layout_attention'] = torch.cat(all_layout_attention, dim=0)
    return result


# ---------------------------------------------------------------------------
# Flatten helpers
# ---------------------------------------------------------------------------

def flatten_valid(logits, labels, mask):
    """Flatten sequence tensors and filter to valid tokens."""
    num_classes = logits.size(-1)
    flat_logits = logits.view(-1, num_classes)
    flat_labels = labels.view(-1)
    flat_mask = mask.view(-1)
    valid = flat_mask.bool() & (flat_labels >= 0)
    return flat_logits[valid], flat_labels[valid], valid


def flatten_layout_attention(layout_attention, mask, valid_mask):
    """Flatten layout_attention to match valid token indices.

    layout_attention may be [batch, total_seq_len] where total_seq_len
    includes visual patch tokens (e.g. 512 text + 197 visual = 709).
    mask is [batch, text_seq_len] (512).  We slice to text portion first.
    """
    if layout_attention is None:
        return None
    text_seq_len = mask.shape[1]
    if layout_attention.dim() == 2 and layout_attention.shape[1] > text_seq_len:
        layout_attention = layout_attention[:, :text_seq_len].contiguous()
    flat_la = layout_attention.reshape(-1)
    if len(flat_la) == len(valid_mask):
        return flat_la[valid_mask]
    return None


# ---------------------------------------------------------------------------
# Baseline F1
# ---------------------------------------------------------------------------

def run_baseline_evaluation(logits, labels, mask, id2label=None):
    """Standard seqeval F1."""
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

    predictions = logits.argmax(dim=-1)
    pred_labels, true_labels = [], []

    for i in range(len(predictions)):
        pred_seq, true_seq = [], []
        for j in range(len(predictions[i])):
            if mask[i][j] == 0 or labels[i][j] < 0:
                continue
            p = predictions[i][j].item()
            t = labels[i][j].item()
            if id2label:
                pred_seq.append(id2label.get(p, f"LABEL_{p}"))
                true_seq.append(id2label.get(t, f"LABEL_{t}"))
            else:
                pred_seq.append(f"LABEL_{p}")
                true_seq.append(f"LABEL_{t}")
        if pred_seq:
            pred_labels.append(pred_seq)
            true_labels.append(true_seq)

    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)

    logger.info(f"Baseline F1: {f1:.4f}, P: {precision:.4f}, R: {recall:.4f}")
    return {'f1': f1, 'precision': precision, 'recall': recall,
            'classification_report': report}


# ---------------------------------------------------------------------------
# Structure-aware conformal evaluation
# ---------------------------------------------------------------------------

def run_structure_aware_conformal(
    cal_logits: torch.Tensor,
    cal_labels: torch.Tensor,
    test_logits: torch.Tensor,
    test_labels: torch.Tensor,
    cal_layout_attn: Optional[torch.Tensor],
    test_layout_attn: Optional[torch.Tensor],
    target_coverages: List[float],
    score_types: List[str],
    device: torch.device,
) -> Dict[str, dict]:
    """Run conformal evaluation with the structure-aware ConformalCalibrator.

    Iterates over (score_type × target_coverage) and returns a nested dict:
        results[score_type][coverage_key] = {coverage, avg_set_size, ...}
    """
    results: Dict[str, dict] = {}

    for score_type in score_types:
        results[score_type] = {}
        for target_cov in target_coverages:
            alpha = 1 - target_cov
            calibrator = ConformalCalibrator(
                alpha=alpha,
                score_type=score_type,
                lambda_layout=0.1,
                lambda_ocr=0.1,
                temperature=1.0,
            ).to(device)

            # Calibrate
            cal_la = cal_layout_attn.to(device) if cal_layout_attn is not None else None
            cal_result = calibrator.calibrate(
                cal_logits.to(device),
                cal_labels.to(device),
                layout_attention=_expand_layout_attn(cal_la, cal_logits.shape[0]),
                optimize_temperature=True,
            )

            # Evaluate coverage on test set
            test_la = test_layout_attn.to(device) if test_layout_attn is not None else None
            coverage_info = calibrator.compute_coverage(
                test_logits.to(device),
                test_labels.to(device),
                layout_attention=_expand_layout_attn(test_la, test_logits.shape[0]),
            )

            # Also get prediction sets for per-class analysis
            pred_results = calibrator.predict(
                test_logits.to(device),
                layout_attention=_expand_layout_attn(test_la, test_logits.shape[0]),
            )

            set_sizes = [r.set_size for r in pred_results]

            cov_key = f"coverage_{int(target_cov * 100)}"
            results[score_type][cov_key] = {
                'target_coverage': target_cov,
                'coverage': coverage_info['coverage'],
                'coverage_gap': coverage_info['coverage_gap'],
                'avg_set_size': coverage_info['avg_set_size'],
                'median_set_size': float(np.median(set_sizes)) if set_sizes else 0,
                'singleton_rate': float(np.mean([1 if s == 1 else 0 for s in set_sizes])) if set_sizes else 0,
                'threshold': calibrator.threshold.item(),
                'temperature': calibrator.temperature.item(),
                'n_cal': len(cal_logits),
                'n_test': len(test_logits),
            }

            logger.info(
                f"  {score_type} @ {target_cov*100:.0f}%: "
                f"cov={coverage_info['coverage']*100:.1f}%, "
                f"set_size={coverage_info['avg_set_size']:.2f}, "
                f"temp={calibrator.temperature.item():.3f}"
            )

    return results


def _expand_layout_attn(layout_attn, n_tokens):
    """Reshape layout_attention for the ConformalCalibrator which expects
    [batch_size, num_regions]. If layout_attn is 1-D (flat tokens), reshape
    to [n_tokens, 1] so entropy computation works."""
    if layout_attn is None:
        return None
    if layout_attn.dim() == 1:
        # Per-token scalar attention → wrap to [n, 1]
        return layout_attn.unsqueeze(-1)
    return layout_attn


# ---------------------------------------------------------------------------
# Softmax-only baseline (uses evaluation/conformal.py)
# ---------------------------------------------------------------------------

def run_softmax_conformal(
    cal_logits, cal_labels, test_logits, test_labels,
    target_coverages, methods=('lac', 'aps'),
):
    """Run the simpler softmax-only conformal evaluation for comparison."""
    num_classes = cal_logits.size(-1)
    results = {}

    for method in methods:
        results[method] = {}
        for target_cov in target_coverages:
            predictor = ConformalPredictor(
                num_classes=num_classes,
                target_coverage=target_cov,
                method=method,
            )
            predictor.calibrate(cal_logits, cal_labels)
            metrics = predictor.evaluate(test_logits, test_labels)
            cov_key = f"coverage_{int(target_cov * 100)}"
            results[method][cov_key] = metrics
            logger.info(
                f"  {method}_softmax @ {target_cov*100:.0f}%: "
                f"cov={metrics['coverage']*100:.1f}%, "
                f"set_size={metrics['avg_set_size']:.2f}"
            )
    return results


# ---------------------------------------------------------------------------
# Per-class coverage
# ---------------------------------------------------------------------------

def compute_per_class_coverage(
    test_logits: torch.Tensor,
    test_labels: torch.Tensor,
    calibrator: ConformalCalibrator,
    test_layout_attn: Optional[torch.Tensor],
    device: torch.device,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """Compute coverage and avg set size per class."""
    pred_results = calibrator.predict(
        test_logits.to(device),
        layout_attention=_expand_layout_attn(
            test_layout_attn.to(device) if test_layout_attn is not None else None,
            test_logits.shape[0]),
    )

    per_class: Dict[int, Dict[str, list]] = defaultdict(lambda: {'covered': [], 'set_sizes': []})
    for i, (res, lbl) in enumerate(zip(pred_results, test_labels.tolist())):
        per_class[lbl]['covered'].append(1 if lbl in res.prediction_set else 0)
        per_class[lbl]['set_sizes'].append(res.set_size)

    summary = {}
    for cls_id in range(num_classes):
        if cls_id in per_class:
            d = per_class[cls_id]
            summary[cls_id] = {
                'coverage': float(np.mean(d['covered'])),
                'avg_set_size': float(np.mean(d['set_sizes'])),
                'n_samples': len(d['covered']),
            }
    return summary


# ---------------------------------------------------------------------------
# Per-token JSON output
# ---------------------------------------------------------------------------

def save_per_token_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    calibrator_lac: ConformalCalibrator,
    calibrator_aps: ConformalCalibrator,
    layout_attn: Optional[torch.Tensor],
    id2label: Optional[Dict],
    output_path: str,
    device: torch.device,
):
    """Save per-document, per-token prediction data to JSON."""
    num_docs = logits.shape[0]
    seq_len = logits.shape[1]
    documents = []

    for doc_idx in range(num_docs):
        doc_logits = logits[doc_idx]  # [seq_len, C]
        doc_labels = labels[doc_idx]  # [seq_len]
        doc_mask = mask[doc_idx]  # [seq_len]
        doc_la = layout_attn[doc_idx] if layout_attn is not None else None

        tokens_data = []
        for t in range(seq_len):
            if doc_mask[t] == 0 or doc_labels[t] < 0:
                continue

            tok_logits = doc_logits[t:t+1].to(device)
            tok_label = doc_labels[t].item()
            probs = F.softmax(tok_logits, dim=-1)
            pred = probs.argmax(dim=-1).item()
            conf = probs[0, pred].item()

            # LAC prediction set
            lac_res = calibrator_lac.predict(
                tok_logits,
                layout_attention=_expand_layout_attn(
                    doc_la[t:t+1].to(device) if doc_la is not None else None, 1),
            )
            # APS prediction set
            aps_res = calibrator_aps.predict(
                tok_logits,
                layout_attention=_expand_layout_attn(
                    doc_la[t:t+1].to(device) if doc_la is not None else None, 1),
            )

            true_label_str = id2label.get(tok_label, str(tok_label)) if id2label else str(tok_label)
            pred_label_str = id2label.get(pred, str(pred)) if id2label else str(pred)

            tokens_data.append({
                'true_label': true_label_str,
                'pred_label': pred_label_str,
                'confidence': round(conf, 6),
                'prediction_set_lac': lac_res[0].prediction_set,
                'set_size_lac': lac_res[0].set_size,
                'prediction_set_aps': aps_res[0].prediction_set,
                'set_size_aps': aps_res[0].set_size,
            })

        if tokens_data:
            documents.append({
                'doc_id': doc_idx,
                'tokens': tokens_data,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(documents, f, indent=2)
    logger.info(f"Saved per-token predictions ({len(documents)} docs) to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CaliDoc Conformal Prediction v2 — Structure-Aware")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='funsd',
                        choices=['funsd', 'cord'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str,
                        default='./results/conformal_v2')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrained_name', type=str,
                        default='microsoft/layoutlmv3-large')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for calibration split (FUNSD)')
    parser.add_argument('--save_per_token', action='store_true',
                        help='Save per-token predictions to JSON')
    parser.add_argument('--use_structure_aware', action='store_true',
                        help='Include structure-aware scoring in ablation')
    parser.add_argument('--calibration_ratio', type=float, default=0.7,
                        help='Fraction of FUNSD test set for calibration '
                             '(ignored for CORD which uses val set)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model and data
    model, processor, loaders, id2label, num_labels = load_model_and_data(
        args.checkpoint_path, args.dataset, args.data_root,
        args.batch_size, args.num_workers, args.pretrained_name,
    )

    target_coverages = [0.90, 0.95, 0.99]

    # ------------------------------------------------------------------
    # Collect predictions
    # ------------------------------------------------------------------
    if args.dataset == 'cord' and 'val' in loaders:
        logger.info("Collecting calibration predictions (val split)...")
        cal_preds = collect_predictions(model, loaders['val'], device)
        logger.info("Collecting test predictions...")
        test_preds = collect_predictions(model, loaders['test'], device)

        cal_logits, cal_labels, cal_valid = flatten_valid(
            cal_preds['logits'], cal_preds['labels'], cal_preds['mask'])
        test_logits, test_labels, test_valid = flatten_valid(
            test_preds['logits'], test_preds['labels'], test_preds['mask'])

        cal_la = flatten_layout_attention(
            cal_preds.get('layout_attention'), cal_preds['mask'], cal_valid)
        test_la = flatten_layout_attention(
            test_preds.get('layout_attention'), test_preds['mask'], test_valid)
    else:
        # FUNSD: split test set
        logger.info("Collecting test predictions...")
        test_preds = collect_predictions(model, loaders['test'], device)

        flat_logits, flat_labels, valid_mask = flatten_valid(
            test_preds['logits'], test_preds['labels'], test_preds['mask'])
        flat_la = flatten_layout_attention(
            test_preds.get('layout_attention'), test_preds['mask'], valid_mask)

        n_samples = len(flat_logits)
        n_cal = int(n_samples * args.calibration_ratio)
        perm = torch.randperm(n_samples)
        cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

        cal_logits = flat_logits[cal_idx]
        cal_labels = flat_labels[cal_idx]
        test_logits = flat_logits[test_idx]
        test_labels = flat_labels[test_idx]
        cal_la = flat_la[cal_idx] if flat_la is not None else None
        test_la = flat_la[test_idx] if flat_la is not None else None

    logger.info(f"Calibration: {len(cal_logits)}, Test: {len(test_logits)}")

    results = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint_path,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'n_cal': len(cal_logits),
        'n_test': len(test_logits),
    }

    # ------------------------------------------------------------------
    # 1. Baseline F1
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE EVALUATION")
    logger.info("=" * 60)
    results['baseline'] = run_baseline_evaluation(
        test_preds['logits'], test_preds['labels'],
        test_preds['mask'], id2label)

    # ------------------------------------------------------------------
    # 2. Calibration analysis
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION ANALYSIS")
    logger.info("=" * 60)
    results['calibration'] = compute_calibration_error(
        test_preds['logits'], test_preds['labels'],
        n_bins=15, mask=test_preds['mask'])
    logger.info(f"ECE: {results['calibration']['ece']*100:.2f}%")

    # ------------------------------------------------------------------
    # 3. Softmax-only conformal (LAC + APS) — baseline
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SOFTMAX-ONLY CONFORMAL (baseline)")
    logger.info("=" * 60)
    results['conformal_softmax'] = run_softmax_conformal(
        cal_logits, cal_labels, test_logits, test_labels,
        target_coverages, methods=('lac', 'aps'))

    # ------------------------------------------------------------------
    # 4. Structure-aware conformal (LAC + APS + structure_aware)
    # ------------------------------------------------------------------
    if args.use_structure_aware:
        logger.info("\n" + "=" * 60)
        logger.info("STRUCTURE-AWARE CONFORMAL")
        logger.info("=" * 60)
        score_types = ['lac', 'aps', 'structure_aware']
        results['conformal_structure_aware'] = run_structure_aware_conformal(
            cal_logits, cal_labels, test_logits, test_labels,
            cal_la, test_la, target_coverages, score_types, device)

    # ------------------------------------------------------------------
    # 5. Temperature scaling baseline
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("TEMPERATURE SCALING BASELINE")
    logger.info("=" * 60)
    temp_calibrator = ConformalCalibrator(
        alpha=0.1, score_type='lac', temperature=1.0,
    ).to(device)
    # Optimize temperature only (no conformal threshold — just check ECE)
    temp_calibrator._optimize_temperature(
        cal_logits.to(device), cal_labels.to(device))
    opt_temp = temp_calibrator.temperature.item()
    logger.info(f"Optimized temperature: {opt_temp:.4f}")

    # Compute ECE with temperature-scaled logits
    scaled_test_logits = test_preds['logits'] / opt_temp
    temp_cal = compute_calibration_error(
        scaled_test_logits, test_preds['labels'],
        n_bins=15, mask=test_preds['mask'])
    results['temperature_scaling'] = {
        'optimized_temperature': opt_temp,
        'ece_before': results['calibration']['ece'],
        'ece_after': temp_cal['ece'],
    }
    logger.info(f"ECE before temp scaling: {results['calibration']['ece']*100:.2f}%")
    logger.info(f"ECE after temp scaling:  {temp_cal['ece']*100:.2f}%")

    # ------------------------------------------------------------------
    # 6. Per-class coverage (structure-aware LAC at 90%)
    # ------------------------------------------------------------------
    if args.use_structure_aware:
        logger.info("\n" + "=" * 60)
        logger.info("PER-CLASS COVERAGE (structure-aware LAC, 90%)")
        logger.info("=" * 60)
        sa_cal = ConformalCalibrator(
            alpha=0.1, score_type='structure_aware',
        ).to(device)
        sa_cal.calibrate(
            cal_logits.to(device), cal_labels.to(device),
            layout_attention=_expand_layout_attn(
                cal_la.to(device) if cal_la is not None else None,
                cal_logits.shape[0]),
            optimize_temperature=True,
        )
        per_class = compute_per_class_coverage(
            test_logits, test_labels, sa_cal, test_la, device, num_labels)
        results['per_class_coverage'] = {
            str(k): v for k, v in per_class.items()}
        for cls_id, info in sorted(per_class.items()):
            label_name = id2label.get(cls_id, str(cls_id)) if id2label else str(cls_id)
            logger.info(f"  {label_name}: cov={info['coverage']*100:.1f}%, "
                        f"set_size={info['avg_set_size']:.2f}, n={info['n_samples']}")

    # ------------------------------------------------------------------
    # 7. Save per-token predictions
    # ------------------------------------------------------------------
    if args.save_per_token and args.use_structure_aware:
        logger.info("\nSaving per-token predictions...")
        # Build calibrators for per-token output
        lac_cal = ConformalCalibrator(
            alpha=0.1, score_type='structure_aware',
        ).to(device)
        lac_cal.calibrate(
            cal_logits.to(device), cal_labels.to(device),
            layout_attention=_expand_layout_attn(
                cal_la.to(device) if cal_la is not None else None,
                cal_logits.shape[0]),
            optimize_temperature=True,
        )
        aps_cal = ConformalCalibrator(
            alpha=0.1, score_type='aps',
        ).to(device)
        aps_cal.calibrate(
            cal_logits.to(device), cal_labels.to(device),
            layout_attention=_expand_layout_attn(
                cal_la.to(device) if cal_la is not None else None,
                cal_logits.shape[0]),
            optimize_temperature=True,
        )
        per_token_path = os.path.join(
            args.output_dir, f'per_token_predictions_{args.dataset}.json')
        save_per_token_predictions(
            test_preds['logits'], test_preds['labels'], test_preds['mask'],
            lac_cal, aps_cal,
            test_preds.get('layout_attention'),
            id2label, per_token_path, device,
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f'results_v2_{timestamp}.json')

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(out_path, 'w') as f:
        json.dump(_convert(results), f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Baseline F1: {results['baseline']['f1']*100:.2f}%")
    logger.info(f"ECE: {results['calibration']['ece']*100:.2f}%")
    if 'temperature_scaling' in results:
        logger.info(f"ECE (temp-scaled): {results['temperature_scaling']['ece_after']*100:.2f}%")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Cal/Test split: {len(cal_logits)}/{len(test_logits)}")


if __name__ == '__main__':
    main()
