#!/usr/bin/env python3
"""
MC Dropout Baseline Evaluation

Addresses:
  - Reviewer weakness #4: missing MC Dropout baseline

MC Dropout (Gal & Ghahramani, 2016):
  1. T=10 forward passes with dropout enabled at test time
  2. Compute predictive entropy from averaged softmax
  3. Compare uncertainty quality: predictive entropy vs CP set sizes
  4. Evaluate calibration of MC Dropout confidence vs softmax confidence

Requires: GPU + model weights

Usage:
    python scripts/evaluate_mc_dropout.py \
        --checkpoint_path outputs/funsd_aug_6631427/best \
        --dataset funsd \
        --data_root data \
        --output_dir results/mc_dropout
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

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
        test_dataset = CORDDataset(
            data_dir=os.path.join(data_root, 'cord'),
            processor=processor, split='test',
            max_length=512, use_huggingface=True)
        loaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers)

    return model, loaders, num_labels


def enable_mc_dropout(model):
    """Enable dropout at test time for MC Dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def collect_mc_predictions(model, dataloader, device, T=10):
    """Collect T forward passes with dropout enabled.

    Returns averaged softmax probabilities and individual pass logits.
    """
    model.eval()
    model.to(device)
    enable_mc_dropout(model)

    # First pass: collect shapes
    all_logits_passes = []

    for t in range(T):
        pass_logits, pass_labels, pass_masks = [], [], []

        for batch in tqdm(dataloader, desc=f"MC pass {t+1}/{T}"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(
                    images=batch.get('pixel_values'),
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    bbox=batch.get('bbox'),
                    labels=batch.get('labels'),
                    return_loss=False,
                )
            pass_logits.append(outputs['logits'].cpu())
            if t == 0:
                pass_labels.append(batch['labels'].cpu())
                pass_masks.append(batch['attention_mask'].cpu())

        all_logits_passes.append(torch.cat(pass_logits, dim=0))
        if t == 0:
            labels = torch.cat(pass_labels, dim=0)
            masks = torch.cat(pass_masks, dim=0)

    # Stack: [T, batch, seq_len, C]
    stacked_logits = torch.stack(all_logits_passes, dim=0)

    # Average softmax
    softmax_passes = F.softmax(stacked_logits, dim=-1)  # [T, B, S, C]
    mean_probs = softmax_passes.mean(dim=0)  # [B, S, C]

    return {
        'mean_probs': mean_probs,
        'stacked_logits': stacked_logits,
        'labels': labels,
        'mask': masks,
        'T': T,
    }


def flatten_valid(tensor, labels, mask):
    """Flatten and filter valid tokens."""
    if tensor.dim() == 3:
        # [B, S, C]
        B, S, C = tensor.shape
        flat = tensor.view(-1, C)
    else:
        flat = tensor.view(-1)
    flat_labels = labels.view(-1)
    flat_mask = mask.view(-1)
    valid = flat_mask.bool() & (flat_labels >= 0)
    if tensor.dim() == 3:
        return flat[valid], flat_labels[valid]
    return flat[valid], flat_labels[valid]


def compute_predictive_entropy(mean_probs):
    """Compute predictive entropy H[y|x] = -sum p(y|x) log p(y|x)."""
    # Clip for numerical stability
    probs = torch.clamp(mean_probs, min=1e-10)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy


def compute_mutual_information(stacked_logits):
    """Compute mutual information (model uncertainty).

    MI = H[y|x] - E_theta[H[y|x,theta]]
    = predictive entropy - mean entropy of individual passes
    """
    T = stacked_logits.shape[0]
    softmax_passes = F.softmax(stacked_logits, dim=-1)

    # Predictive entropy
    mean_probs = softmax_passes.mean(dim=0)
    pred_entropy = compute_predictive_entropy(mean_probs)

    # Mean entropy across passes
    per_pass_entropy = []
    for t in range(T):
        probs_t = torch.clamp(softmax_passes[t], min=1e-10)
        ent_t = -torch.sum(probs_t * torch.log(probs_t), dim=-1)
        per_pass_entropy.append(ent_t)

    mean_entropy = torch.stack(per_pass_entropy, dim=0).mean(dim=0)
    mutual_info = pred_entropy - mean_entropy

    return pred_entropy, mutual_info


def compute_metrics(mean_probs, labels, stacked_logits=None):
    """Compute MC Dropout metrics."""
    # Prediction accuracy
    preds = mean_probs.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()

    # MC confidence (max of averaged softmax)
    mc_confidence = mean_probs.max(dim=-1).values

    # Predictive entropy
    pred_entropy = compute_predictive_entropy(mean_probs)

    # Standard softmax confidence (single pass)
    if stacked_logits is not None:
        single_probs = F.softmax(stacked_logits[0], dim=-1)
        single_confidence = single_probs.max(dim=-1).values
        single_preds = single_probs.argmax(dim=-1)
        single_accuracy = (single_preds == labels).float().mean().item()
    else:
        single_confidence = mc_confidence
        single_accuracy = accuracy

    # Calibration: bin by confidence and compute accuracy
    def compute_ece(confidence, correct, n_bins=15):
        boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidence > boundaries[i]) & (confidence <= boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = correct[mask].float().mean().item()
                bin_conf = confidence[mask].mean().item()
                ece += mask.sum().item() / len(confidence) * abs(bin_acc - bin_conf)
        return ece

    correct = (preds == labels)
    mc_ece = compute_ece(mc_confidence, correct)

    single_correct = (single_preds == labels) if stacked_logits is not None else correct
    single_ece = compute_ece(single_confidence, single_correct)

    # Uncertainty-error correlation
    is_error = ~correct
    entropy_errors = pred_entropy[is_error].mean().item() if is_error.sum() > 0 else 0
    entropy_correct = pred_entropy[correct].mean().item() if correct.sum() > 0 else 0

    return {
        'mc_accuracy': accuracy,
        'single_accuracy': single_accuracy,
        'mc_ece': mc_ece,
        'single_ece': single_ece,
        'mean_predictive_entropy': pred_entropy.mean().item(),
        'mean_mc_confidence': mc_confidence.mean().item(),
        'mean_single_confidence': single_confidence.mean().item() if stacked_logits is not None else None,
        'entropy_on_errors': entropy_errors,
        'entropy_on_correct': entropy_correct,
        'entropy_separation': entropy_errors - entropy_correct,
        'n_tokens': len(labels),
        'error_rate': is_error.float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="MC Dropout Baseline Evaluation")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='funsd',
                        choices=['funsd', 'cord'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str,
                        default='./results/mc_dropout')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--T', type=int, default=10,
                        help='Number of MC Dropout forward passes')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, loaders, num_labels = load_model_and_data(
        args.checkpoint_path, args.dataset, args.data_root,
        args.batch_size, args.num_workers)

    logger.info(f"Running MC Dropout with T={args.T} passes...")
    mc_preds = collect_mc_predictions(
        model, loaders['test'], device, T=args.T)

    # Flatten to valid tokens
    mean_probs, labels = flatten_valid(
        mc_preds['mean_probs'], mc_preds['labels'], mc_preds['mask'])

    # Also flatten stacked logits
    T = mc_preds['T']
    B, S, C = mc_preds['stacked_logits'].shape[1:]
    flat_mask = mc_preds['mask'].view(-1)
    flat_labels = mc_preds['labels'].view(-1)
    valid = flat_mask.bool() & (flat_labels >= 0)

    stacked_flat = mc_preds['stacked_logits'].view(T, -1, C)[:, valid, :]

    logger.info(f"Valid tokens: {len(labels)}")

    # Compute metrics
    metrics = compute_metrics(mean_probs, labels, stacked_flat)

    # Compute mutual information
    pred_entropy, mutual_info = compute_mutual_information(stacked_flat)
    metrics['mean_mutual_information'] = mutual_info[valid[:len(mutual_info)]].mean().item() if len(mutual_info) > 0 else 0

    logger.info(f"\nResults:")
    logger.info(f"  MC accuracy: {metrics['mc_accuracy']*100:.2f}%")
    logger.info(f"  Single-pass accuracy: {metrics['single_accuracy']*100:.2f}%")
    logger.info(f"  MC ECE: {metrics['mc_ece']*100:.2f}%")
    logger.info(f"  Single-pass ECE: {metrics['single_ece']*100:.2f}%")
    logger.info(f"  Mean predictive entropy: {metrics['mean_predictive_entropy']:.4f}")
    logger.info(f"  Entropy on errors: {metrics['entropy_on_errors']:.4f}")
    logger.info(f"  Entropy on correct: {metrics['entropy_on_correct']:.4f}")
    logger.info(f"  Entropy separation: {metrics['entropy_separation']:.4f}")

    results = {
        'dataset': args.dataset,
        'T': args.T,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        **metrics,
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f'mc_dropout_{args.dataset}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"MC DROPOUT RESULTS: {args.dataset.upper()} (T={args.T})")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    for key in ['mc_accuracy', 'single_accuracy', 'mc_ece', 'single_ece',
                'mean_predictive_entropy', 'entropy_on_errors',
                'entropy_on_correct', 'entropy_separation']:
        val = metrics[key]
        if 'accuracy' in key or 'ece' in key:
            print(f"{key:<35} {val*100:>9.2f}%")
        else:
            print(f"{key:<35} {val:>10.4f}")


if __name__ == '__main__':
    main()
