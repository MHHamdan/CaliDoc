#!/usr/bin/env python3
"""
RAPS (Regularized Adaptive Prediction Sets) Baseline Evaluation

Addresses:
  - Reviewer weakness #4: missing RAPS baseline

RAPS score = APS score + λ × max(0, rank - k_reg)
where rank is the position of label y in the sorted probability vector.
The regularization penalty encourages smaller prediction sets.

Requires: GPU + model weights (needs full softmax vectors via inference)

Usage:
    python scripts/evaluate_raps.py \
        --checkpoint_path outputs/funsd_aug_6631427/best \
        --dataset funsd \
        --data_root data \
        --output_dir results/raps
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_data(checkpoint_path, dataset_name, data_root,
                        batch_size=8, num_workers=4,
                        pretrained_name='microsoft/layoutlmv3-large'):
    """Load model and datasets."""
    from transformers import AutoProcessor
    from calidoc.models import CaliDocModel
    from torch.utils.data import DataLoader

    num_labels = 7 if dataset_name == 'funsd' else 53
    encoder_config = {
        'model_name': pretrained_name,
        'task': 'ner',
        'num_labels': num_labels,
    }
    model = CaliDocModel.from_pretrained(
        checkpoint_path,
        encoder_config=encoder_config,
        use_quality_routing=False,
        use_calibration=False,
    )
    processor = AutoProcessor.from_pretrained(pretrained_name, apply_ocr=False)

    loaders = {}
    if dataset_name == 'funsd':
        from calidoc.data.funsd_dataset import FUNSDDataset
        test_dataset = FUNSDDataset(
            data_dir=os.path.join(data_root, 'funsd'),
            processor=processor, split='test',
            max_length=512, use_huggingface=True)
        loaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers)
    else:
        from calidoc.data.cord_dataset import CORDDataset
        val_dataset = CORDDataset(
            data_dir=os.path.join(data_root, 'cord'),
            processor=processor, split='validation',
            max_length=512, use_huggingface=True)
        test_dataset = CORDDataset(
            data_dir=os.path.join(data_root, 'cord'),
            processor=processor, split='test',
            max_length=512, use_huggingface=True)
        loaders['val'] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers)
        loaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers)

    return model, loaders, num_labels


def collect_predictions(model, dataloader, device):
    """Collect full softmax vectors."""
    model.eval()
    model.to(device)
    all_logits, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
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
            all_logits.append(outputs['logits'].cpu())
            all_labels.append(batch['labels'].cpu())
            all_masks.append(batch['attention_mask'].cpu())

    return {
        'logits': torch.cat(all_logits, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'mask': torch.cat(all_masks, dim=0),
    }


def flatten_valid(logits, labels, mask):
    """Flatten and filter valid tokens."""
    num_classes = logits.size(-1)
    flat_logits = logits.view(-1, num_classes)
    flat_labels = labels.view(-1)
    flat_mask = mask.view(-1)
    valid = flat_mask.bool() & (flat_labels >= 0)
    return flat_logits[valid], flat_labels[valid]


def compute_raps_scores(logits, labels, lambda_reg=0.01, k_reg=3):
    """Compute RAPS nonconformity scores.

    RAPS score for label y:
      s(x, y) = sum_{j: pi_j >= pi_y} pi_j + lambda * max(0, rank(y) - k_reg) + u * pi_y
    where pi is the sorted probability vector and u ~ Uniform(0,1) for tie-breaking.

    Args:
        logits: [n, C] logits
        labels: [n] true labels
        lambda_reg: regularization strength
        k_reg: regularization threshold (rank below which no penalty)

    Returns:
        scores: [n] RAPS scores
    """
    probs = F.softmax(logits, dim=-1)
    n, C = probs.shape

    # Sort probabilities in decreasing order
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

    # Find rank of true label (1-indexed)
    ranks = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        rank_mask = sorted_indices[i] == labels[i]
        ranks[i] = rank_mask.nonzero(as_tuple=True)[0].item() + 1

    # Cumulative sum of sorted probabilities up to rank of true label
    cumsum = sorted_probs.cumsum(dim=-1)
    aps_scores = torch.zeros(n)
    for i in range(n):
        r = ranks[i].item()
        aps_scores[i] = cumsum[i, r - 1].item()

    # RAPS regularization
    reg_penalty = lambda_reg * torch.clamp(ranks.float() - k_reg, min=0)

    # Random tie-breaking
    u = torch.rand(n)
    true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    raps_scores = aps_scores + reg_penalty - u * true_probs

    return raps_scores.numpy()


def compute_lac_scores(logits, labels):
    """Compute LAC nonconformity scores."""
    probs = F.softmax(logits, dim=-1)
    true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return (1 - true_probs).numpy()


def compute_aps_scores(logits, labels):
    """Compute APS nonconformity scores."""
    probs = F.softmax(logits, dim=-1)
    n, C = probs.shape
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    scores = torch.zeros(n)
    for i in range(n):
        rank_mask = sorted_indices[i] == labels[i]
        rank = rank_mask.nonzero(as_tuple=True)[0].item()
        scores[i] = cumsum[i, rank].item()

    u = torch.rand(n)
    true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return (scores - u * true_probs).numpy()


def calibrate_and_evaluate(cal_scores, test_scores, test_logits, test_labels,
                           alpha=0.10):
    """Calibrate threshold and evaluate coverage + set sizes."""
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    threshold = float(np.quantile(cal_scores, q_level))

    # Coverage
    covered = test_scores <= threshold
    coverage = float(np.mean(covered))

    # Prediction set sizes
    probs = F.softmax(test_logits, dim=-1).numpy()
    set_sizes = []
    for i in range(len(test_logits)):
        # For LAC-style: classes with prob >= 1-threshold
        if 'lac' in str(type(cal_scores)):
            n_included = int(np.sum(probs[i] >= 1 - threshold))
        else:
            # For APS/RAPS: accumulate sorted probs until score exceeds threshold
            sorted_p = np.sort(probs[i])[::-1]
            cumsum = np.cumsum(sorted_p)
            n_included = int(np.searchsorted(cumsum, threshold) + 1)
            n_included = min(n_included, len(sorted_p))
        set_sizes.append(n_included)

    return {
        'coverage': coverage,
        'threshold': threshold,
        'avg_set_size': float(np.mean(set_sizes)),
        'median_set_size': float(np.median(set_sizes)),
        'singleton_rate': float(np.mean([1 if s == 1 else 0 for s in set_sizes])),
        'n_cal': len(cal_scores),
        'n_test': len(test_scores),
    }


def run_comparison(cal_logits, cal_labels, test_logits, test_labels,
                   target_coverages, lambda_values, k_reg_values):
    """Run LAC vs APS vs RAPS comparison."""
    results = {}

    for alpha_inv in target_coverages:
        alpha = 1 - alpha_inv
        cov_key = f"coverage_{int(alpha_inv * 100)}"
        results[cov_key] = {}

        # LAC
        cal_lac = compute_lac_scores(cal_logits, cal_labels)
        test_lac = compute_lac_scores(test_logits, test_labels)
        lac_result = calibrate_and_evaluate(
            cal_lac, test_lac, test_logits, test_labels, alpha)
        results[cov_key]['lac'] = lac_result
        logger.info(f"  LAC @ {alpha_inv*100:.0f}%: cov={lac_result['coverage']*100:.1f}%, "
                     f"set_size={lac_result['avg_set_size']:.2f}")

        # APS
        cal_aps = compute_aps_scores(cal_logits, cal_labels)
        test_aps = compute_aps_scores(test_logits, test_labels)
        aps_result = calibrate_and_evaluate(
            cal_aps, test_aps, test_logits, test_labels, alpha)
        results[cov_key]['aps'] = aps_result
        logger.info(f"  APS @ {alpha_inv*100:.0f}%: cov={aps_result['coverage']*100:.1f}%, "
                     f"set_size={aps_result['avg_set_size']:.2f}")

        # RAPS with various hyperparameters
        for lam in lambda_values:
            for k in k_reg_values:
                cal_raps = compute_raps_scores(
                    cal_logits, cal_labels, lambda_reg=lam, k_reg=k)
                test_raps = compute_raps_scores(
                    test_logits, test_labels, lambda_reg=lam, k_reg=k)
                raps_result = calibrate_and_evaluate(
                    cal_raps, test_raps, test_logits, test_labels, alpha)
                raps_key = f"raps_lam{lam}_k{k}"
                results[cov_key][raps_key] = raps_result
                logger.info(
                    f"  RAPS(λ={lam},k={k}) @ {alpha_inv*100:.0f}%: "
                    f"cov={raps_result['coverage']*100:.1f}%, "
                    f"set_size={raps_result['avg_set_size']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="RAPS Baseline Evaluation")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='funsd',
                        choices=['funsd', 'cord'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str,
                        default='./results/raps')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_values', nargs='+', type=float,
                        default=[0.001, 0.01, 0.1])
    parser.add_argument('--k_reg_values', nargs='+', type=int,
                        default=[1, 3, 5])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, loaders, num_labels = load_model_and_data(
        args.checkpoint_path, args.dataset, args.data_root,
        args.batch_size, args.num_workers)

    target_coverages = [0.90, 0.95, 0.99]

    if args.dataset == 'cord' and 'val' in loaders:
        logger.info("Collecting calibration (val) predictions...")
        cal_preds = collect_predictions(model, loaders['val'], device)
        logger.info("Collecting test predictions...")
        test_preds = collect_predictions(model, loaders['test'], device)
        cal_logits, cal_labels = flatten_valid(
            cal_preds['logits'], cal_preds['labels'], cal_preds['mask'])
        test_logits, test_labels = flatten_valid(
            test_preds['logits'], test_preds['labels'], test_preds['mask'])
    else:
        logger.info("Collecting test predictions...")
        test_preds = collect_predictions(model, loaders['test'], device)
        flat_logits, flat_labels = flatten_valid(
            test_preds['logits'], test_preds['labels'], test_preds['mask'])
        n = len(flat_logits)
        perm = torch.randperm(n)
        n_cal = int(n * 0.7)
        cal_logits = flat_logits[perm[:n_cal]]
        cal_labels = flat_labels[perm[:n_cal]]
        test_logits = flat_logits[perm[n_cal:]]
        test_labels = flat_labels[perm[n_cal:]]

    logger.info(f"Cal: {len(cal_logits)}, Test: {len(test_logits)}")

    results = run_comparison(
        cal_logits, cal_labels, test_logits, test_labels,
        target_coverages, args.lambda_values, args.k_reg_values)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f'raps_results_{args.dataset}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"RAPS COMPARISON: {args.dataset.upper()}")
    print(f"{'='*80}")
    for cov_key in sorted(results.keys()):
        print(f"\n--- {cov_key} ---")
        print(f"{'Method':<25} {'Coverage':>8} {'Set Size':>9} {'Singleton':>9}")
        print("-" * 55)
        for method_key in sorted(results[cov_key].keys()):
            r = results[cov_key][method_key]
            print(f"{method_key:<25} {r['coverage']*100:>7.1f}% "
                  f"{r['avg_set_size']:>9.2f} "
                  f"{r['singleton_rate']*100:>8.1f}%")


if __name__ == '__main__':
    main()
