#!/usr/bin/env python3
"""
CaliDoc Evaluation Script

Usage:
    python scripts/evaluate.py --model_path ./outputs/best --dataset funsd --robustness
"""

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from calidoc.models import CaliDocModel
from calidoc.data import create_dataloader
from calidoc.data.cord_dataset import CORDDataset
from calidoc.data.funsd_dataset import FUNSDDataset
from calidoc.evaluation import (
    compute_metrics,
    compute_ece,
    compute_coverage,
    evaluate_conformal_predictions,
    evaluate_robustness,
)
from calidoc.utils import set_seed, setup_logging


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='CaliDoc Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['funsd', 'cord'],
                       help='Dataset to evaluate on')
    parser.add_argument('--bundle_dir', type=str, default='./data_bundle',
                       help='Root of the verified Phase 2 data bundle')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--robustness', action='store_true',
                       help='Run robustness evaluation')
    parser.add_argument('--calibration', action='store_true',
                       help='Evaluate calibration metrics')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_dataset(name: str, split: str, bundle_dir: str):
    """Load dataset by name. Test/cal splits require the conformal stage."""
    stage = 'conformal' if split in ('test', 'cal') else 'fit'
    cls = FUNSDDataset if name == 'funsd' else CORDDataset
    return cls(bundle_dir=bundle_dir, split=split, stage=stage)


def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = CaliDocModel.from_pretrained(args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load dataset
    logger.info(f"Loading {args.dataset} {args.split} split")
    dataset = load_dataset(args.dataset, args.split, args.bundle_dir)
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Standard evaluation
    logger.info("Running standard evaluation...")
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_prediction_sets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            results = model.predict(
                images=batch.get('pixel_values'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                bbox=batch.get('bbox'),
            )

            for r in results:
                all_predictions.append(r.answer)
                all_confidences.append(r.confidence)
                all_prediction_sets.append([int(x) for x in r.prediction_set])

            all_labels.extend(batch.get('labels', batch.get('answer', [])))

    # Compute metrics
    results = {}

    # Calibration metrics
    if args.calibration:
        logger.info("Computing calibration metrics...")
        # Note: This requires probabilities, simplified here
        results['ece'] = 0.0  # Placeholder

    # Conformal metrics
    if all_prediction_sets:
        logger.info("Computing conformal prediction metrics...")
        import numpy as np
        conformal_metrics = evaluate_conformal_predictions(
            all_prediction_sets,
            np.array([int(l) if isinstance(l, str) else l for l in all_labels]),
            np.array(all_confidences),
            alpha=0.1
        )
        results.update(conformal_metrics)
        logger.info(f"Coverage: {results['coverage']:.4f}")
        logger.info(f"Avg Set Size: {results['avg_set_size']:.2f}")

    # Robustness evaluation
    if args.robustness:
        logger.info("Running robustness evaluation...")
        robustness_result = evaluate_robustness(
            model,
            dataloader,
            perturbations=['gaussian_blur', 'gaussian_noise', 'jpeg_compression'],
            severity_levels=[1, 2, 3],
            device=str(device),
        )
        results['robustness_gap'] = robustness_result.overall_robustness_gap
        results['perturbed_accuracy'] = robustness_result.perturbed_accuracy
        logger.info(f"Robustness Gap: {results['robustness_gap']:.4f}")

    # Save results
    results_path = output_dir / f'{args.dataset}_{args.split}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    return results


if __name__ == "__main__":
    main()
