#!/usr/bin/env python3
"""
E1 (primary arm) -- planted label noise with FULL fine-tuning.

The frozen-feature sweep in scripts/noise_dose_response.py is not sufficient on
its own for two reasons, both of which suppress the effect we are measuring:

  1. its trunk was fine-tuned on CLEAN labels, so the representation already
     encodes the correct structure that the injected noise is meant to destroy;
  2. a lightly-trained linear probe averages noise out instead of memorising
     it, and memorisation is precisely the mechanism by which label noise
     corrupts conformal calibration.

This script therefore corrupts the TRAIN split and fine-tunes the whole model
from the pretrained checkpoint, exactly as the paper's main runs do. Everything
downstream -- temperature, calibration quantiles, evaluation -- uses CLEAN
labels, so any coverage change is attributable to what the model learned.

One cell = (noise_model, eps, seed). Cells are independent; run them serially.

Usage:
  python scripts/noise_full_finetune.py --dataset cord --bundle_dir ./data_bundle \
      --noise_model class_targeted --eps 0.05 --seed 0 \
      --out_dir results/local_20260803/noise_ft
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from calidoc.data import create_dataloader
from calidoc.data.bundle import FUNSD_CALIB_SEEDS, conformal_split
from calidoc.models import CaliDocModel
from calidoc.training import CaliDocTrainer
from calidoc.training.trainer import TrainingConfig
from calidoc.utils import set_seed

from ablations_phase2 import DATASETS, collect
from conditional_cp_baselines import (
    classwise_qhat, conditional_summary, evaluate_qhat, marginal_qhat)
from evaluate_conformal_phase2 import fit_temperature, probs_of
from noise_dose_response import (
    pick_target_and_confusable,
    inject_bio_shift, inject_class_targeted, inject_uniform)

logger = logging.getLogger(__name__)

ALPHA = 0.10

# Pinned to the Phase 2 recipe so noise is the only variable.
RECIPE = {
    'cord':        dict(epochs=30, lr=2e-5, patience=5,  label_smoothing=0.0, llrd=1.0),
    'wildreceipt': dict(epochs=30, lr=2e-5, patience=5,  label_smoothing=0.0, llrd=1.0),
    'sroie':       dict(epochs=30, lr=2e-5, patience=5,  label_smoothing=0.0, llrd=1.0),
    'funsd':       dict(epochs=120, lr=5e-5, patience=15, label_smoothing=0.1, llrd=0.9),
}


def corrupt_train_split(train_ds, noise_model, eps, seed, id2label, label2id,
                        num_labels, target=None, confusable=None):
    """Rewrite train_ds.df['ner_tags'] in place with planted noise.
    Returns (n_flipped, flipped_mask_per_doc, target, confusable)."""
    docs = [np.array([label2id[t] for t in tags]) for tags in train_ds.df['ner_tags']]
    rng = np.random.RandomState(1000 * seed + int(eps * 1000))

    if noise_model == 'uniform':
        noisy, flipped = inject_uniform(docs, eps, num_labels, rng)
    elif noise_model == 'class_targeted':
        if target is None:
            target, confusable = pick_target_and_confusable(docs, id2label)
        noisy, flipped = inject_class_targeted(docs, eps, target, confusable, rng)
    elif noise_model == 'bio_shift':
        noisy, flipped = inject_bio_shift(docs, eps, id2label, label2id, rng)
    else:
        raise ValueError(noise_model)

    train_ds.df = train_ds.df.copy()
    train_ds.df['ner_tags'] = [[id2label[int(c)] for c in y] for y in noisy]
    return int(sum(f.sum() for f in flipped)), flipped, target, confusable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--bundle_dir', required=True)
    ap.add_argument('--noise_model', required=True,
                    choices=['uniform', 'class_targeted', 'bio_shift'])
    ap.add_argument('--eps', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model_name', default='microsoft/layoutlmv3-large')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--keep_checkpoint', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cell = f'{args.dataset}_{args.noise_model}_eps{args.eps}_s{args.seed}'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'{cell}.json'
    if out_json.exists():
        logger.info("%s already exists, skipping", out_json)
        return

    set_seed(args.seed)
    cls, es_split = DATASETS[args.dataset]
    bundle_dir = Path(args.bundle_dir)
    device = torch.device(args.device)
    rec = RECIPE[args.dataset]

    train_ds = cls(bundle_dir, split='train', stage='fit')
    es_ds = cls(bundle_dir, split=es_split, stage='fit')
    id2label = train_ds.id2label
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    n_flipped, _, target, confusable = corrupt_train_split(
        train_ds, args.noise_model, args.eps, args.seed,
        id2label, label2id, num_labels)
    logger.info("cell %s: flipped %d / %d train tokens", cell, n_flipped,
                sum(len(t) for t in train_ds.df['ner_tags']))

    model = CaliDocModel(
        encoder_type='layoutlmv3',
        encoder_config={'task': 'ner', 'num_labels': num_labels,
                        'model_name': args.model_name},
        use_quality_routing=False, use_calibration=False)

    ckpt_dir = out_dir / 'ckpt' / cell
    cfg = TrainingConfig(
        epochs=rec['epochs'], learning_rate=rec['lr'], weight_decay=0.01,
        warmup_ratio=0.1, batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        task_weight=1.0, calibration_weight=0.0, quality_weight=0.0,
        routing_weight=0.0, output_dir=str(ckpt_dir), mixed_precision=True,
        device=str(device), use_wandb=False, task='ner', id2label=id2label,
        llrd_rate=rec['llrd'], early_stopping_patience=rec['patience'])
    cfg.label_smoothing = rec['label_smoothing']

    trainer = CaliDocTrainer(
        model=model,
        train_dataloader=create_dataloader(train_ds, batch_size=args.batch_size,
                                           shuffle=True),
        eval_dataloader=create_dataloader(es_ds, batch_size=args.batch_size,
                                          shuffle=False),
        config=cfg)
    trainer.train()

    # ---- evaluate under the clean protocol ------------------------------
    model.to(device)
    es_pd = collect(model, es_ds, device, args.batch_size)
    temperature = fit_temperature(torch.cat([l for l, _, _ in es_pd]),
                                  torch.cat([y for _, y, _ in es_pd]))

    if args.dataset == 'cord':
        cal_ds = cls(bundle_dir, split='cal', stage='conformal')
        ev_ds = cls(bundle_dir, split='test', stage='conformal')
    else:
        cal_ids, ev_ids = conformal_split(bundle_dir, args.dataset,
                                          FUNSD_CALIB_SEEDS[0])
        cal_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=cal_ids)
        ev_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=ev_ids)

    def pack(ds):
        pd_ = collect(model, ds, device, args.batch_size)
        return (np.concatenate([probs_of(l, temperature) for l, _, _ in pd_]),
                np.concatenate([y.numpy() for _, y, _ in pd_]))

    cal_probs, cal_labels = pack(cal_ds)
    ev_probs, ev_labels = pack(ev_ds)
    classes = sorted(set(np.unique(cal_labels)) | set(np.unique(ev_labels)))

    qcw = classwise_qhat(cal_probs, cal_labels, ALPHA, classes)
    cw = evaluate_qhat(ev_probs, ev_labels, qcw, id2label, classes)
    qm = {c: marginal_qhat(cal_probs, cal_labels, ALPHA) for c in classes}
    mg = evaluate_qhat(ev_probs, ev_labels, qm, id2label, classes)

    from seqeval.metrics import f1_score
    true_seqs, pred_seqs = [], []
    for logits, labels, _ in collect(model, ev_ds, device, args.batch_size):
        preds = probs_of(logits, temperature).argmax(axis=1)
        y = labels.numpy()
        true_seqs.append([id2label[int(t)] for t in y])
        pred_seqs.append([id2label[int(p)] for p in preds])

    tgt_name = id2label[int(target)] if target is not None else None
    result = {
        'cell': cell, 'dataset': args.dataset, 'noise_model': args.noise_model,
        'eps': args.eps, 'seed': args.seed,
        'arm': 'full_finetune',
        'n_flipped_train_tokens': n_flipped,
        'class_targeted_map': (f'{tgt_name} -> {id2label[int(confusable)]}'
                               if args.noise_model == 'class_targeted' else None),
        'temperature': temperature,
        'test_f1': float(f1_score(true_seqs, pred_seqs)),
        'token_accuracy': float((ev_probs.argmax(1) == ev_labels).mean()),
        'alpha': ALPHA,
        'marginal': {'coverage': mg['marginal_coverage'],
                     'avg_set_size': mg['avg_set_size'],
                     **conditional_summary(mg['per_class'], ALPHA)},
        'classwise': {'coverage': cw['marginal_coverage'],
                      'avg_set_size': cw['avg_set_size'],
                      **conditional_summary(cw['per_class'], ALPHA)},
        'target_class': tgt_name,
        'target_class_coverage_classwise': (
            cw['per_class'].get(tgt_name, {}).get('coverage') if tgt_name else None),
        'target_class_coverage_marginal': (
            mg['per_class'].get(tgt_name, {}).get('coverage') if tgt_name else None),
        'per_class_marginal': mg['per_class'],
        'per_class_classwise': cw['per_class'],
    }
    out_json.write_text(json.dumps(result, indent=2))
    logger.info("cell %s: F1 %.4f marg %.4f CovGap %.2fpp target %s",
                cell, result['test_f1'], result['marginal']['coverage'],
                result['marginal']['covgap_pp'],
                result['target_class_coverage_classwise'])

    if not args.keep_checkpoint:
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        logger.info("removed %s", ckpt_dir)


if __name__ == '__main__':
    main()
