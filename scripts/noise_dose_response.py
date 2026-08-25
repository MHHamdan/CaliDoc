#!/usr/bin/env python3
"""
E1 -- planted label noise at known rates -> per-class conformal coverage.

Turns the "per-class conformal coverage audits labels" claim from an anecdote
into a calibrated instrument: we corrupt a CLEAN dataset at known rates under
three noise models chosen to mimic the two real failures we observed, then
measure how per-class coverage responds.

  uniform         each token independently relabelled to a uniformly random
                  DIFFERENT class with probability eps. The generic control.
  class_targeted  tokens of ONE chosen class are relabelled to a single fixed
                  confusable class with probability eps. Mimics the CORD
                  label-schema bug, where a whole field family was silently
                  folded into the wrong label.
  bio_shift       entity-span boundaries are shifted by one token with
                  probability eps (B->I of the running span, or the first I
                  promoted to B). Mimics the SROIE fuzzy-alignment noise.

Efficiency: encoder features do NOT depend on labels, so the LayoutLMv3 trunk
is run ONCE per dataset and cached; each (noise model, eps, seed) cell then
trains only the linear token-classification head on those frozen features.
That is the same head the full model uses, but it cannot memorise noise the
way full fine-tuning does, so the sweep UNDERSTATES the effect. Run
--full_finetune at a few cells as the control arm; do not report the frozen
sweep alone.

Ground truth of which tokens were corrupted is written out for E2
(the Cleanlab / AUM / Data Maps head-to-head).

Usage:
  python scripts/noise_dose_response.py --dataset cord --bundle_dir ./data_bundle \
      --model_path outputs/cord_retrain_s42_local_20260803/best \
      --output results/local_20260803/noise_dose_cord.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from calidoc.data import create_dataloader
from calidoc.data.bundle import FUNSD_CALIB_SEEDS, conformal_split
from calidoc.models import CaliDocModel

from ablations_phase2 import DATASETS
from conditional_cp_baselines import (
    classwise_qhat, conditional_summary, evaluate_qhat, marginal_qhat)
from evaluate_conformal_phase2 import conformal_quantile, lac_scores

logger = logging.getLogger(__name__)

EPS_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
NOISE_MODELS = ('uniform', 'class_targeted', 'bio_shift')
SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.10
HEAD_EPOCHS = 40
HEAD_LR = 1e-3


# --------------------------------------------------------------------------
# Noise models (operate on integer label sequences, per document)
# --------------------------------------------------------------------------

def inject_uniform(docs, eps, num_labels, rng):
    out, flipped = [], []
    for y in docs:
        y2 = y.copy()
        m = rng.random_sample(len(y)) < eps
        for i in np.where(m)[0]:
            choices = [c for c in range(num_labels) if c != y[i]]
            y2[i] = choices[rng.randint(len(choices))]
        out.append(y2); flipped.append(m)
    return out, flipped


def pick_target_and_confusable(docs, id2label):
    """Choose the class-targeted corruption as a CROSS-FIELD confusion.

    The real CORD defect folded a whole field family into a different label, so
    the target is the most frequent B- class and the confusable is the most
    frequent B- class of a DIFFERENT field. Mapping B-X onto I-X instead would
    only move an entity boundary, which is what bio_shift already models -- the
    two noise models must stay distinct or the dose-response curves are not
    interpretable.
    """
    all_y = np.concatenate(docs)
    counts = {int(c): int((all_y == c).sum()) for c in np.unique(all_y)}
    b_classes = [c for c in counts if id2label[c].startswith('B-')]
    if not b_classes:
        ranked = sorted(counts, key=counts.get, reverse=True)
        return int(ranked[0]), int(ranked[1])

    target = max(b_classes, key=lambda c: counts[c])
    target_field = id2label[target][2:]
    others = [c for c in b_classes if id2label[c][2:] != target_field]
    if others:
        confusable = max(others, key=lambda c: counts[c])
    else:
        confusable = int([c for c in sorted(counts, key=counts.get, reverse=True)
                          if c != target][0])
    return int(target), int(confusable)


def inject_class_targeted(docs, eps, target, confusable, rng):
    out, flipped = [], []
    for y in docs:
        y2 = y.copy()
        m = (y == target) & (rng.random_sample(len(y)) < eps)
        y2[m] = confusable
        out.append(y2); flipped.append(m)
    return out, flipped


def inject_bio_shift(docs, eps, id2label, label2id, rng):
    """Shift entity-span boundaries by one token."""
    out, flipped = [], []
    for y in docs:
        y2 = y.copy()
        m = np.zeros(len(y), bool)
        tags = [id2label[int(t)] for t in y]
        for i, t in enumerate(tags):
            if not t.startswith('B-') or rng.random_sample() >= eps:
                continue
            typ = t[2:]
            i_tag = label2id.get(f'I-{typ}')
            if i_tag is None:
                continue
            if i + 1 < len(y) and tags[i + 1] == f'I-{typ}':
                # move the boundary right: B becomes I, next I becomes B
                b_tag = label2id[f'B-{typ}']
                y2[i] = i_tag; y2[i + 1] = b_tag
                m[i] = m[i + 1] = True
            elif i > 0:
                # absorb the preceding token into this span
                y2[i] = i_tag
                m[i] = True
        out.append(y2); flipped.append(m)
    return out, flipped


# --------------------------------------------------------------------------
# Feature caching (encoder is label-independent -> run once)
# --------------------------------------------------------------------------

def encode_split(model, dataset, device, batch_size, labels_only=False):
    """Per-doc (hidden_states[n, H], labels[n]) for tokens with label != -100.

    labels_only skips retaining the hidden states, for callers that need only
    the label sequences and their document boundaries."""
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    feats, labs = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            out = model(images=batch['pixel_values'], input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'], bbox=batch['bbox'],
                        labels=None, return_loss=False)
            h = out['hidden_states'].float().cpu()
            y = batch['labels'].cpu()
            n_text = y.shape[1]
            for i in range(h.shape[0]):
                m = y[i] != -100
                if not labels_only:
                    feats.append(h[i][:n_text][m].numpy().astype(np.float32))
                labs.append(y[i][m].numpy().astype(np.int64))
    return feats, labs


def train_head(feats, labels, num_labels, device, epochs=HEAD_EPOCHS,
               lr=HEAD_LR, seed=0):
    """Train the linear token-classification head on frozen features."""
    torch.manual_seed(seed)
    X = torch.from_numpy(np.concatenate(feats)).to(device)
    y = torch.from_numpy(np.concatenate(labels)).to(device)
    head = torch.nn.Linear(X.shape[1], num_labels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(X), y)
        loss.backward()
        opt.step()
    return head


def head_probs(head, feats, device, temperature=1.0):
    with torch.no_grad():
        X = torch.from_numpy(np.concatenate(feats)).to(device)
        return F.softmax(head(X) / temperature, dim=-1).cpu().numpy()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--bundle_dir', required=True)
    ap.add_argument('--model_path', required=True,
                    help='checkpoint whose trunk provides the frozen features')
    ap.add_argument('--output', required=True)
    ap.add_argument('--flip_out', default=None,
                    help='where to write corrupted-token ground truth for E2')
    ap.add_argument('--eps', default=','.join(str(e) for e in EPS_GRID))
    ap.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    eps_grid = [float(x) for x in args.eps.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    cls, es_split = DATASETS[args.dataset]
    bundle_dir = Path(args.bundle_dir)
    device = torch.device(args.device)

    mp = Path(args.model_path)
    cfg = json.loads((mp / 'config.json').read_text())
    model = CaliDocModel(
        encoder_type=cfg['encoder_type'], encoder_config=cfg['encoder_config'],
        use_quality_routing=cfg.get('use_quality_routing', False),
        use_calibration=cfg.get('use_calibration', False))
    model.load_state_dict(torch.load(mp / 'pytorch_model.bin', map_location='cpu',
                                     weights_only=False))
    model.to(device)

    train_ds = cls(bundle_dir, split='train', stage='fit')
    id2label = train_ds.id2label
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    logger.info("Encoding splits once (features are label-independent)")
    tr_feats, tr_labels = encode_split(model, train_ds, device, args.batch_size)
    if args.dataset == 'cord':
        cal_ds = cls(bundle_dir, split='cal', stage='conformal')
        ev_ds = cls(bundle_dir, split='test', stage='conformal')
    else:
        cal_ids, ev_ids = conformal_split(bundle_dir, args.dataset, FUNSD_CALIB_SEEDS[0])
        cal_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=cal_ids)
        ev_ds = cls(bundle_dir, split='test', stage='conformal', doc_ids=ev_ids)
    cal_feats, cal_labels_l = encode_split(model, cal_ds, device, args.batch_size)
    ev_feats, ev_labels_l = encode_split(model, ev_ds, device, args.batch_size)
    cal_labels = np.concatenate(cal_labels_l)
    ev_labels = np.concatenate(ev_labels_l)
    classes = sorted(set(np.unique(cal_labels)) | set(np.unique(ev_labels)))

    # Pick the target class for class_targeted noise: the most frequent
    # non-O B- class in train, mapped onto its own I- partner if present,
    # else onto the most frequent other class. Mimics one field family being
    # silently folded into a neighbouring label.
    tr_all = np.concatenate(tr_labels)
    target, confusable = pick_target_and_confusable(tr_labels, id2label)
    logger.info("class_targeted: %s -> %s", id2label[int(target)],
                id2label[int(confusable)])

    cells, flip_records = [], []
    for model_name in NOISE_MODELS:
        for eps in eps_grid:
            for seed in seeds:
                if eps == 0.0 and seed != seeds[0]:
                    continue                     # eps=0 is deterministic
                rng = np.random.RandomState(1000 * seed + int(eps * 1000))
                if model_name == 'uniform':
                    noisy, flipped = inject_uniform(tr_labels, eps, num_labels, rng)
                elif model_name == 'class_targeted':
                    noisy, flipped = inject_class_targeted(
                        tr_labels, eps, target, confusable, rng)
                else:
                    noisy, flipped = inject_bio_shift(
                        tr_labels, eps, id2label, label2id, rng)

                head = train_head(tr_feats, noisy, num_labels, device, seed=seed)
                cal_probs = head_probs(head, cal_feats, device)
                ev_probs = head_probs(head, ev_feats, device)

                qcw = classwise_qhat(cal_probs, cal_labels, ALPHA, classes)
                cw = evaluate_qhat(ev_probs, ev_labels, qcw, id2label, classes)
                qm = {c: marginal_qhat(cal_probs, cal_labels, ALPHA) for c in classes}
                mg = evaluate_qhat(ev_probs, ev_labels, qm, id2label, classes)

                tgt_name = id2label[int(target)]
                cell = {
                    'noise_model': model_name, 'eps': eps, 'seed': seed,
                    'n_flipped': int(sum(f.sum() for f in flipped)),
                    'n_train_tokens': int(len(tr_all)),
                    'accuracy': float((ev_probs.argmax(1) == ev_labels).mean()),
                    'marginal': {
                        'coverage': mg['marginal_coverage'],
                        'avg_set_size': mg['avg_set_size'],
                        **conditional_summary(mg['per_class'], ALPHA)},
                    'classwise': {
                        'coverage': cw['marginal_coverage'],
                        'avg_set_size': cw['avg_set_size'],
                        **conditional_summary(cw['per_class'], ALPHA)},
                    'target_class': tgt_name,
                    'target_class_coverage_classwise':
                        cw['per_class'].get(tgt_name, {}).get('coverage'),
                    'target_class_coverage_marginal':
                        mg['per_class'].get(tgt_name, {}).get('coverage'),
                }
                cells.append(cell)
                logger.info("%s eps=%.2f seed=%d | acc %.3f marg %.3f "
                            "CovGap %.2fpp target %.3f",
                            model_name, eps, seed, cell['accuracy'],
                            cell['marginal']['coverage'],
                            cell['marginal']['covgap_pp'] or float('nan'),
                            cell['target_class_coverage_classwise']
                            if cell['target_class_coverage_classwise'] is not None
                            else float('nan'))

                if args.flip_out and model_name == 'class_targeted':
                    flip_records.append({
                        'eps': eps, 'seed': seed,
                        'flipped_per_doc': [f.tolist() for f in flipped],
                        'noisy_labels_per_doc': [y.tolist() for y in noisy],
                    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'dataset': args.dataset,
        'trunk_checkpoint': str(mp),
        'protocol': {
            'alpha': ALPHA,
            'training': ('linear head on FROZEN encoder features; the trunk is '
                         'label-independent so it is encoded once. This '
                         'UNDERSTATES noise memorisation relative to full '
                         'fine-tuning -- see the --full_finetune control arm.'),
            'head_epochs': HEAD_EPOCHS, 'head_lr': HEAD_LR,
            'noise_models': list(NOISE_MODELS),
            'class_targeted_map': f'{id2label[int(target)]} -> {id2label[int(confusable)]}',
            'eps_grid': eps_grid, 'seeds': seeds,
            'evaluation': 'per-class coverage measured on CLEAN calibration and test labels',
        },
        'cells': cells,
    }, indent=2))
    logger.info("Wrote %s", out)

    if args.flip_out:
        fo = Path(args.flip_out)
        fo.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fo,
            train_features=np.concatenate(tr_feats),
            train_docstart=np.concatenate([[0], np.cumsum([len(y) for y in tr_labels])]),
            clean_labels=tr_all,
            records=json.dumps(flip_records))
        logger.info("Wrote %s", fo)


if __name__ == '__main__':
    main()
