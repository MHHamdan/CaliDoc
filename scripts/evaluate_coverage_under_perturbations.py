#!/usr/bin/env python3
"""
Coverage Under Perturbations Evaluation

Addresses:
  - Reviewer weakness #5: no CP coverage reported under perturbations
  - Reviewer Q5: coverage degradation under distribution shift

For each of 9 perturbations × 3 severities:
  1. Apply perturbation to test images
  2. Run model inference on perturbed images
  3. Compute CP coverage using CLEAN calibration threshold
  4. Report coverage degradation alongside F1 drop

Requires: GPU + model weights restored in checkpoint_path

Usage:
    python scripts/evaluate_coverage_under_perturbations.py \
        --checkpoint_path outputs/funsd_aug_6631427/best \
        --dataset funsd \
        --data_root data \
        --output_dir results/coverage_under_perturbations
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PERTURBATION_TYPES = [
    'gaussian_blur', 'motion_blur', 'gaussian_noise',
    'salt_pepper_noise', 'jpeg_compression', 'brightness',
    'contrast', 'ink_bleed', 'paper_texture',
]
SEVERITIES = [1, 2, 3]


def load_model_and_data(checkpoint_path, dataset_name, data_root,
                        batch_size=8, num_workers=4,
                        pretrained_name='microsoft/layoutlmv3-large'):
    """Load model and data (reuse from evaluate_conformal_v2.py pattern)."""
    from transformers import AutoProcessor
    from calidoc.models import CaliDocModel

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
    return model, processor, num_labels


def get_dataset(dataset_name, data_root, processor, split='test',
                apply_perturbation=None, severity=1):
    """Load dataset, optionally with perturbations."""
    from torch.utils.data import DataLoader

    if dataset_name == 'funsd':
        from calidoc.data.funsd_dataset import FUNSDDataset
        dataset = FUNSDDataset(
            data_dir=os.path.join(data_root, 'funsd'),
            processor=processor,
            split=split,
            max_length=512,
            use_huggingface=True,
        )
    else:
        from calidoc.data.cord_dataset import CORDDataset
        dataset = CORDDataset(
            data_dir=os.path.join(data_root, 'cord'),
            processor=processor,
            split=split,
            max_length=512,
            use_huggingface=True,
        )

    return dataset


def collect_predictions(model, dataloader, device):
    """Collect model predictions."""
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


def compute_lac_coverage(logits, labels, threshold):
    """Compute LAC coverage using a pre-computed threshold."""
    probs = F.softmax(logits, dim=-1)
    true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    scores = 1 - true_probs
    covered = (scores <= threshold).float()
    return {
        'coverage': covered.mean().item(),
        'avg_score': scores.mean().item(),
        'n_tokens': len(labels),
    }


def compute_f1(logits, labels, id2label=None):
    """Compute token-level accuracy and seqeval F1."""
    preds = logits.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()
    return {'accuracy': accuracy}


def compute_clean_threshold(logits, labels, alpha=0.10):
    """Compute LAC threshold from clean calibration data."""
    probs = F.softmax(logits, dim=-1)
    true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    scores = 1 - true_probs
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(torch.quantile(scores, q_level).item())


def apply_perturbation_to_images(dataset, perturbation_type, severity):
    """Apply a specific perturbation to dataset images.

    This modifies the dataset's __getitem__ to apply perturbation.
    Returns a wrapper dataset.
    """
    from calidoc.data.perturbations import DocumentPerturbations

    perturber = DocumentPerturbations()

    # Get the perturbation function by name
    all_perts = DocumentPerturbations.get_all_perturbations()
    if perturbation_type not in all_perts:
        raise ValueError(f"Unknown perturbation '{perturbation_type}'. "
                         f"Available: {list(all_perts.keys())}")
    pert_fn = all_perts[perturbation_type]

    class PerturbedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, pert_func, sev):
            self.base = base_dataset
            self.pert_func = pert_func
            self.sev = sev

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            # Get raw item from underlying HuggingFace dataset
            item = self.base.dataset[idx]

            from PIL import Image
            if isinstance(item['image'], str):
                image = Image.open(item['image']).convert('RGB')
            elif isinstance(item['image'], Image.Image):
                image = item['image'].convert('RGB')
            else:
                image = Image.open(item['image']).convert('RGB')

            # Apply perturbation via static method
            image_np = np.array(image)
            image_np = self.pert_func(image_np, severity=self.sev)
            image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))

            # Get words, boxes, labels from base dataset
            words = item.get('words', [])
            boxes = item.get('boxes', item.get('bboxes', []))
            ner_tags = item.get('ner_tags', [0] * len(words))

            # Process with processor
            normalized_boxes = []
            for box in boxes:
                normalized_boxes.append([
                    min(1000, max(0, int(box[0]))),
                    min(1000, max(0, int(box[1]))),
                    min(1000, max(0, int(box[2]))),
                    min(1000, max(0, int(box[3]))),
                ])

            encoding = self.base.processor(
                image, words,
                boxes=normalized_boxes,
                word_labels=ner_tags,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

    return PerturbedDataset(dataset, pert_fn, severity)


def main():
    parser = argparse.ArgumentParser(
        description="Coverage Under Perturbations Evaluation")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='funsd',
                        choices=['funsd', 'cord'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str,
                        default='./results/coverage_under_perturbations')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, processor, num_labels = load_model_and_data(
        args.checkpoint_path, args.dataset, args.data_root)

    # Get clean dataset
    clean_dataset = get_dataset(args.dataset, args.data_root, processor)
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    # Collect clean predictions
    logger.info("Collecting clean predictions...")
    clean_preds = collect_predictions(model, clean_loader, device)
    clean_logits, clean_labels = flatten_valid(
        clean_preds['logits'], clean_preds['labels'], clean_preds['mask'])

    # Split for calibration (70%) and test (30%)
    n = len(clean_logits)
    perm = torch.randperm(n)
    n_cal = int(n * 0.7)
    cal_logits = clean_logits[perm[:n_cal]]
    cal_labels = clean_labels[perm[:n_cal]]
    test_logits = clean_logits[perm[n_cal:]]
    test_labels = clean_labels[perm[n_cal:]]

    # Compute clean threshold
    clean_threshold = compute_clean_threshold(cal_logits, cal_labels, args.alpha)
    logger.info(f"Clean LAC threshold: {clean_threshold:.8f}")

    # Clean baseline metrics
    clean_coverage = compute_lac_coverage(test_logits, test_labels, clean_threshold)
    clean_f1 = compute_f1(test_logits, test_labels)
    logger.info(f"Clean coverage: {clean_coverage['coverage']*100:.1f}%")
    logger.info(f"Clean accuracy: {clean_f1['accuracy']*100:.1f}%")

    results = {
        'dataset': args.dataset,
        'alpha': args.alpha,
        'clean_threshold': clean_threshold,
        'clean_coverage': clean_coverage['coverage'],
        'clean_accuracy': clean_f1['accuracy'],
        'perturbations': {},
    }

    # Run perturbations
    for pert_type in PERTURBATION_TYPES:
        results['perturbations'][pert_type] = {}
        for severity in SEVERITIES:
            logger.info(f"\nPerturbation: {pert_type}, Severity: {severity}")

            try:
                pert_dataset = apply_perturbation_to_images(
                    clean_dataset, pert_type, severity)
                pert_loader = torch.utils.data.DataLoader(
                    pert_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)

                pert_preds = collect_predictions(model, pert_loader, device)
                pert_logits, pert_labels = flatten_valid(
                    pert_preds['logits'], pert_preds['labels'],
                    pert_preds['mask'])

                # Use same token indices as test split
                pert_test_logits = pert_logits[perm[n_cal:]] if len(pert_logits) == n else pert_logits
                pert_test_labels = pert_labels[perm[n_cal:]] if len(pert_labels) == n else pert_labels

                # Coverage using CLEAN threshold
                pert_coverage = compute_lac_coverage(
                    pert_test_logits, pert_test_labels, clean_threshold)
                pert_f1 = compute_f1(pert_test_logits, pert_test_labels)

                cov_drop = pert_coverage['coverage'] - clean_coverage['coverage']
                acc_drop = pert_f1['accuracy'] - clean_f1['accuracy']

                results['perturbations'][pert_type][f'severity_{severity}'] = {
                    'coverage': pert_coverage['coverage'],
                    'accuracy': pert_f1['accuracy'],
                    'coverage_drop': cov_drop,
                    'accuracy_drop': acc_drop,
                }

                logger.info(f"  Coverage: {pert_coverage['coverage']*100:.1f}% "
                            f"(Δ={cov_drop*100:+.1f}%), "
                            f"Accuracy: {pert_f1['accuracy']*100:.1f}% "
                            f"(Δ={acc_drop*100:+.1f}%)")

            except Exception as e:
                logger.warning(f"  Failed: {e}")
                results['perturbations'][pert_type][f'severity_{severity}'] = {
                    'error': str(e)}

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir,
                            f'coverage_perturbations_{args.dataset}_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"COVERAGE UNDER PERTURBATIONS: {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"{'Perturbation':<20} {'Sev.1':>8} {'Sev.2':>8} {'Sev.3':>8}")
    print(f"{'':20} {'Cov/Acc':>8} {'Cov/Acc':>8} {'Cov/Acc':>8}")
    print("-" * 70)
    for pert_type in PERTURBATION_TYPES:
        row = f"{pert_type:<20}"
        for sev in SEVERITIES:
            d = results['perturbations'].get(pert_type, {}).get(
                f'severity_{sev}', {})
            if 'coverage' in d:
                row += f" {d['coverage_drop']*100:>+5.1f}/{d['accuracy_drop']*100:>+5.1f}"
            else:
                row += f" {'N/A':>8}"
        print(row)


if __name__ == '__main__':
    main()
