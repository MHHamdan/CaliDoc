#!/usr/bin/env python3
"""
E2 -- benchmark per-class conformal coverage against standard label-error
detectors, under planted noise where the corrupted tokens are known.

Detectors compared (all consume the same out-of-fold predicted probabilities,
so no method gets a better model than any other):

  cleanlab        cleanlab.token_classification.filter.find_label_issues
                  (Wang et al., arXiv:2210.03920) -- the closest published
                  detector to ours: BIO, token level, probability-only.
  max_softmax     1 - p(given label). The trivial baseline.
  icp_pvalue      per-sample inductive-CP p-value for the GIVEN label
                  (Zhan et al., PLOS Comput Biol 2025) -- conformal, but
                  per-sample rather than class-conditional.
  mondrian_gap    ours: each token inherits the class-conditional coverage
                  DEFICIT of its given label. Deliberately class-granular.
  aum             Area Under the Margin (Pleiss et al., NeurIPS 2020), from
                  training dynamics.
  datamap_conf    dataset cartography confidence, and
  datamap_var     variability (Swayamdipta et al., EMNLP 2020).

Two model fits are needed and they are not interchangeable:
  * out-of-fold probabilities (K-fold over DOCUMENTS) for the probability-only
    detectors, because Cleanlab and CP both assume out-of-sample predictions;
  * a full-data fit recording per-epoch margins for AUM / Data Maps, which are
    defined on training dynamics.

Model choice is what makes or breaks this benchmark, and we got it wrong once.
Fitting only a linear probe on FROZEN pretrained features drives every
detector to chance (AUPRC 0.011-0.032 at 1.10% prevalence, precision@n = 0
across the board on CORD): pretrained LayoutLMv3 features simply do not
linearly separate semantically adjacent fields such as B-MENU.NM and
B-MENU.PRICE, so there is no signal for any method to rank. We therefore
fine-tune the last K encoder layers (default 4) on the NOISY labels, which is
the setting Cleanlab and confident learning actually assume -- the model must
fit the corrupted data well enough that its disagreements are informative.

Starting from a checkpoint fine-tuned on CLEAN labels would be the opposite
error: every detector would be scored on features that already encode the
right answers. So the trunk starts pretrained and only ever sees noisy labels.
`--finetune_layers 0` restores the (inadequate) linear-probe behaviour and is
retained only to document the negative result. The out-of-fold accuracy against
the noisy labels is recorded in the output so the model's competence is
auditable rather than assumed.

Usage:
  python scripts/label_error_baselines.py --dataset cord \
      --bundle_dir ./data_bundle --noise_model class_targeted --eps 0.10 \
      --output results/local_20260803/label_error_baselines_cord.json
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

from calidoc.models import CaliDocModel

from ablations_phase2 import DATASETS
from evaluate_conformal_phase2 import conformal_quantile, lac_scores
from noise_dose_response import (
    encode_split, inject_bio_shift, inject_class_targeted, inject_uniform)

logger = logging.getLogger(__name__)

K_FOLDS = 5
HEAD_EPOCHS = 60
HEAD_LR = 1e-3
ALPHA = 0.10


def train_head_dynamics(X, y, num_labels, device, epochs=HEAD_EPOCHS,
                        lr=HEAD_LR, seed=0, record=False):
    """Train the linear head; optionally record per-epoch gold-label prob and
    margin for every example (needed by AUM and Data Maps)."""
    torch.manual_seed(seed)
    head = torch.nn.Linear(X.shape[1], num_labels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    probs_hist, margin_hist = [], []
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if record:
            with torch.no_grad():
                lg = head(X)
                p = F.softmax(lg, dim=-1)
                gold = p[torch.arange(len(y)), y]
                other = lg.clone()
                other[torch.arange(len(y)), y] = -1e9
                margin = lg[torch.arange(len(y)), y] - other.max(dim=1).values
                probs_hist.append(gold.cpu().numpy())
                margin_hist.append(margin.cpu().numpy())
    return head, (np.array(probs_hist), np.array(margin_hist)) if record else None


def _set_trainable(model, n_layers):
    """Freeze everything except the last n_layers encoder blocks + classifier.

    A linear probe on frozen PRETRAINED features cannot separate semantically
    adjacent fields (e.g. B-MENU.NM vs B-MENU.PRICE), which drives every
    detector to chance and makes the benchmark vacuous. Partially fine-tuning
    gives the detectors a competent model without the cost of five full
    fine-tunes, and without exposing them to clean labels."""
    for p in model.parameters():
        p.requires_grad = False
    enc = model.encoder.encoder.encoder if hasattr(model.encoder, 'encoder') else None
    blocks = None
    for attr in ('layer', 'layers'):
        if enc is not None and hasattr(enc, attr):
            blocks = getattr(enc, attr)
            break
    if blocks is None:                      # fall back: unfreeze by name
        for name, p in model.named_parameters():
            if 'classifier' in name or any(f'.{i}.' in name for i in
                                           range(24 - n_layers, 24)):
                p.requires_grad = True
    else:
        for blk in list(blocks)[-n_layers:]:
            for p in blk.parameters():
                p.requires_grad = True
    for name, p in model.named_parameters():
        if 'classifier' in name:
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def finetune_fold(make_model, train_docs, noisy_docs, device, batch_size,
                  n_layers, epochs, lr, seed):
    """Fine-tune the last n_layers on NOISY labels for one fold."""
    from calidoc.data import create_dataloader
    model = make_model()
    params = _set_trainable(model, n_layers)
    model.to(device).train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    torch.manual_seed(seed)
    ds = train_docs
    ds.df = ds.df.copy()
    ds.df['ner_tags'] = noisy_docs
    loader = create_dataloader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            out = model(images=batch['pixel_values'], input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'], bbox=batch['bbox'],
                        labels=batch['labels'], return_loss=True)
            loss = out['loss'] if 'loss' in out else out['task_loss']
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()
    return model


def encode_split_probs(model, dataset, device, batch_size):
    """Per-doc softmax probabilities for tokens with label != -100."""
    from calidoc.data import create_dataloader
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    per_doc, labs = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            out = model(images=batch['pixel_values'], input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'], bbox=batch['bbox'],
                        labels=None, return_loss=False)
            lg = out['logits'].float().cpu()
            y = batch['labels'].cpu()
            for i in range(lg.shape[0]):
                m = y[i] != -100
                per_doc.append(F.softmax(lg[i][m], dim=-1).numpy().astype(np.float32))
                labs.append(y[i][m].numpy())
    return per_doc, labs


def full_fit_dynamics(make_model, dataset, device, batch_size, n_layers,
                      epochs, lr, seed, flat_labels, starts):
    """One full-data fine-tune on noisy labels, recording the gold-label
    probability and margin for every token after each epoch. AUM and Data Maps
    are defined on exactly these training dynamics."""
    from calidoc.data import create_dataloader
    model = make_model()
    params = _set_trainable(model, n_layers)
    model.to(device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    torch.manual_seed(seed)
    train_loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)

    gold_hist, margin_hist = [], []
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            out = model(images=batch['pixel_values'], input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'], bbox=batch['bbox'],
                        labels=batch['labels'], return_loss=True)
            loss = out['loss'] if 'loss' in out else out['task_loss']
            loss.backward()
            opt.step()
            opt.zero_grad()

        per_doc, _ = encode_split_probs(model, dataset, device, batch_size)
        probs = np.concatenate(per_doc)
        gold = probs[np.arange(len(flat_labels)), flat_labels]
        other = probs.copy()
        other[np.arange(len(flat_labels)), flat_labels] = -1.0
        gold_hist.append(gold)
        margin_hist.append(gold - other.max(axis=1))
    del model
    torch.cuda.empty_cache()
    return np.array(gold_hist), np.array(margin_hist)


def out_of_fold_probs(feats_per_doc, labels_per_doc, num_labels, device,
                      k=K_FOLDS, seed=0):
    """K-fold linear probing over DOCUMENTS (never tokens: tokens within a
    document are not exchangeable, and splitting them would leak).

    Used only when --finetune_layers 0; see finetune_fold otherwise."""
    n_docs = len(feats_per_doc)
    rng = np.random.RandomState(seed)
    fold_of = rng.permutation(n_docs) % k
    lens = [len(y) for y in labels_per_doc]
    total = int(sum(lens))
    out = np.zeros((total, num_labels), dtype=np.float32)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(int)

    for f in range(k):
        tr = [i for i in range(n_docs) if fold_of[i] != f]
        te = [i for i in range(n_docs) if fold_of[i] == f]
        if not te or not tr:
            continue
        Xtr = torch.from_numpy(np.concatenate([feats_per_doc[i] for i in tr])).to(device)
        ytr = torch.from_numpy(np.concatenate([labels_per_doc[i] for i in tr])).to(device)
        head, _ = train_head_dynamics(Xtr, ytr, num_labels, device, seed=seed + f)
        with torch.no_grad():
            for i in te:
                Xi = torch.from_numpy(feats_per_doc[i]).to(device)
                p = F.softmax(head(Xi), dim=-1).cpu().numpy()
                out[starts[i]:starts[i + 1]] = p
    return out, starts


def score_cleanlab(probs, labels_per_doc, starts):
    from cleanlab.token_classification.filter import find_label_issues
    pred_probs = [probs[starts[i]:starts[i + 1]] for i in range(len(labels_per_doc))]
    given = [list(map(int, y)) for y in labels_per_doc]
    issues = find_label_issues(given, pred_probs)
    # find_label_issues returns (sentence_idx, token_idx) pairs, ranked most
    # -likely-issue first. Convert the ranking into a per-token score.
    scores = np.zeros(starts[-1], dtype=np.float64)
    n = len(issues)
    for rank, (si, ti) in enumerate(issues):
        scores[starts[si] + ti] = n - rank
    return scores


def score_icp_pvalue(probs, labels, cal_mask):
    """Per-sample inductive-CP p-value for the GIVEN label: the fraction of
    calibration scores at least as extreme. Low p-value => suspicious label."""
    s_all = lac_scores(probs, labels)
    s_cal = np.sort(s_all[cal_mask])
    n = len(s_cal)
    ranks = np.searchsorted(s_cal, s_all, side='left')
    pvals = (n - ranks + 1) / (n + 1)
    return -pvals            # higher score = more suspicious


def score_mondrian_gap(probs, labels, cal_mask, alpha=ALPHA):
    """Ours, at CLASS granularity: compute class-conditional coverage on a
    held-out portion, then give every token the coverage DEFICIT of its own
    given class. Tokens of a broken class all score alike -- that is the point,
    and the reason this is not expected to win a token-level ranking metric."""
    classes = np.unique(labels)
    cal_probs, cal_labels = probs[cal_mask], labels[cal_mask]
    ev = ~cal_mask
    deficit = {}
    for c in classes:
        cm = cal_labels == c
        em = ev & (labels == c)
        if not cm.any() or not em.any():
            deficit[c] = 0.0
            continue
        q = conformal_quantile(lac_scores(cal_probs[cm], cal_labels[cm]), alpha)
        in_set = probs[em] >= (1.0 - q)
        cov = in_set[np.arange(em.sum()), labels[em]].mean()
        deficit[c] = float(max(0.0, (1 - alpha) - cov))
    return np.array([deficit[c] for c in labels])


def class_level_ranking(scores, flipped, labels, id2label):
    """Class-level localisation: aggregate each detector's token scores to a
    per-class mean, rank classes, and report where the truly-corrupted class
    lands.

    This is the metric that matches the claim -- "which class or schema is
    broken" -- and it is applied IDENTICALLY to every detector, including the
    token-level ones, so nothing is handicapped. Token-level AUPRC cannot test
    this claim, because a class-granular score assigns every token of a class
    the same value and its within-class ordering is arbitrary by construction.

    The corrupted class is defined as the class holding the most planted
    errors (for class-targeted noise this is unambiguous).
    """
    if flipped.sum() == 0:
        return {'note': 'no planted errors'}
    # which GIVEN-label class received the corruption
    corrupted = {}
    for c in np.unique(labels[flipped]):
        corrupted[int(c)] = int((labels[flipped] == c).sum())
    true_class = max(corrupted, key=corrupted.get)

    per_class = {}
    for c in np.unique(labels):
        m = labels == c
        per_class[int(c)] = float(scores[m].mean())
    ranked = sorted(per_class, key=lambda c: -per_class[c])
    rank = ranked.index(true_class) + 1
    return {
        'true_corrupted_class': id2label[true_class],
        'n_errors_in_that_class': corrupted[true_class],
        'rank_of_corrupted_class': rank,
        'n_classes': len(ranked),
        'top_ranked_class': id2label[ranked[0]],
        'reciprocal_rank': 1.0 / rank,
        'hit_at_1': rank == 1,
        'hit_at_3': rank <= 3,
    }


def prf_auprc(scores, flipped):
    from sklearn.metrics import average_precision_score
    order = np.argsort(-scores)
    n_pos = int(flipped.sum())
    if n_pos == 0:
        return {'auprc': None, 'note': 'no planted errors'}
    hits = flipped[order]
    at_k = {}
    for k in (n_pos, 2 * n_pos):
        k = min(k, len(order))
        tp = int(hits[:k].sum())
        at_k[f'top_{k}'] = {'precision': tp / k, 'recall': tp / n_pos}
    return {
        'auprc': float(average_precision_score(flipped, scores)),
        'prevalence': float(flipped.mean()),
        'lift_over_random': (float(average_precision_score(flipped, scores))
                             / float(flipped.mean())),
        'at_k': at_k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--bundle_dir', required=True)
    ap.add_argument('--noise_model', default='class_targeted',
                    choices=['uniform', 'class_targeted', 'bio_shift'])
    ap.add_argument('--eps', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', required=True)
    ap.add_argument('--trunk', default='microsoft/layoutlmv3-large',
                    help='pretrained by default; pass a fine-tuned dir to ablate')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--finetune_layers', type=int, default=4,
                    help='unfreeze the last K encoder layers (0 = linear probe '
                         'on frozen features, which is too weak to be a fair '
                         'benchmark -- see module docstring)')
    ap.add_argument('--ft_epochs', type=int, default=4)
    ap.add_argument('--ft_lr', type=float, default=3e-5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cls, _ = DATASETS[args.dataset]
    bundle_dir = Path(args.bundle_dir)
    device = torch.device(args.device)

    train_ds = cls(bundle_dir, split='train', stage='fit')
    id2label = train_ds.id2label
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    if Path(args.trunk).is_dir():
        cfg = json.loads((Path(args.trunk) / 'config.json').read_text())
        model = CaliDocModel(encoder_type=cfg['encoder_type'],
                             encoder_config=cfg['encoder_config'],
                             use_quality_routing=False, use_calibration=False)
        model.load_state_dict(torch.load(Path(args.trunk) / 'pytorch_model.bin',
                                         map_location='cpu', weights_only=False))
        trunk_kind = f'fine-tuned ({args.trunk})'
    else:
        model = CaliDocModel(
            encoder_type='layoutlmv3',
            encoder_config={'task': 'ner', 'num_labels': num_labels,
                            'model_name': args.trunk},
            use_quality_routing=False, use_calibration=False)
        trunk_kind = f'pretrained ({args.trunk}), no clean-label exposure'
    model.to(device)
    logger.info("trunk: %s", trunk_kind)

    # The frozen-feature path needs encoded features; the fine-tuning path does
    # not, and encoding 800 documents for nothing is pure waste.
    if args.finetune_layers > 0:
        _, clean_labels = encode_split(model, train_ds, device, args.batch_size,
                                       labels_only=True)
        feats = None
        model.to('cpu'); del model; torch.cuda.empty_cache()
    else:
        feats, clean_labels = encode_split(model, train_ds, device, args.batch_size)

    rng = np.random.RandomState(1000 * args.seed + int(args.eps * 1000))
    if args.noise_model == 'uniform':
        noisy, flipped = inject_uniform(clean_labels, args.eps, num_labels, rng)
        target = None
    elif args.noise_model == 'class_targeted':
        all_y = np.concatenate(clean_labels)
        counts = {int(c): int((all_y == c).sum()) for c in np.unique(all_y)}
        b = [c for c in counts if id2label[c].startswith('B-')]
        target = max(b, key=lambda c: counts[c]) if b else max(counts, key=counts.get)
        typ = id2label[target][2:]
        conf = label2id.get(f'I-{typ}')
        if conf is None or conf == target:
            conf = int(sorted(counts, key=counts.get, reverse=True)[1])
        noisy, flipped = inject_class_targeted(clean_labels, args.eps, target, conf, rng)
    else:
        noisy, flipped = inject_bio_shift(clean_labels, args.eps, id2label, label2id, rng)
        target = None

    flat_flipped = np.concatenate(flipped)
    flat_noisy = np.concatenate(noisy)
    logger.info("planted %d / %d token errors (%.2f%%)", flat_flipped.sum(),
                len(flat_flipped), 100 * flat_flipped.mean())

    lens = [len(y) for y in noisy]
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    noisy_tags = [[id2label[int(c)] for c in y] for y in noisy]
    doc_ids = list(train_ds.doc_ids)

    if args.finetune_layers > 0:
        logger.info("out-of-fold probabilities via %d-fold fine-tuning of the "
                    "last %d encoder layers on NOISY labels",
                    K_FOLDS, args.finetune_layers)
        rngf = np.random.RandomState(args.seed)
        fold_of = rngf.permutation(len(doc_ids)) % K_FOLDS
        tag_of = dict(zip(doc_ids, noisy_tags))

        def make_model():
            m = CaliDocModel(
                encoder_type='layoutlmv3',
                encoder_config={'task': 'ner', 'num_labels': num_labels,
                                'model_name': args.trunk},
                use_quality_routing=False, use_calibration=False)
            return m

        def subset(ids):
            ds = cls(bundle_dir, split='train', stage='fit', doc_ids=ids)
            ds.df = ds.df.copy()
            ds.df['ner_tags'] = [tag_of[d] for d in ds.doc_ids]
            return ds

        oof = np.zeros((int(starts[-1]), num_labels), dtype=np.float32)
        pos = {d: i for i, d in enumerate(doc_ids)}
        for f in range(K_FOLDS):
            tr_ids = [d for i, d in enumerate(doc_ids) if fold_of[i] != f]
            te_ids = [d for i, d in enumerate(doc_ids) if fold_of[i] == f]
            if not tr_ids or not te_ids:
                continue
            logger.info("  fold %d/%d: %d train / %d held-out docs",
                        f + 1, K_FOLDS, len(tr_ids), len(te_ids))
            m = finetune_fold(make_model, subset(tr_ids), [tag_of[d] for d in tr_ids],
                              device, args.batch_size, args.finetune_layers,
                              args.ft_epochs, args.ft_lr, args.seed + f)
            te_feats, te_labels = encode_split_probs(m, subset(te_ids), device,
                                                     args.batch_size)
            for d, p in zip(te_ids, te_feats):
                i = pos[d]
                oof[starts[i]:starts[i + 1]] = p
            del m
            torch.cuda.empty_cache()

        acc = float((oof.argmax(1) == flat_noisy).mean())
        logger.info("out-of-fold accuracy against NOISY labels: %.4f", acc)

        logger.info("full-data fine-tune with per-epoch dynamics (AUM / Data Maps)")
        gold_hist, margin_hist = full_fit_dynamics(
            make_model, subset(doc_ids), device, args.batch_size,
            args.finetune_layers, args.ft_epochs, args.ft_lr, args.seed,
            flat_noisy, starts)
    else:
        logger.info("computing out-of-fold probabilities (%d folds, linear probe)",
                    K_FOLDS)
        oof, starts = out_of_fold_probs(feats, noisy, num_labels, device,
                                        seed=args.seed)
        acc = float((oof.argmax(1) == flat_noisy).mean())
        logger.info("out-of-fold accuracy against NOISY labels: %.4f", acc)
        X = torch.from_numpy(np.concatenate(feats)).to(device)
        y = torch.from_numpy(flat_noisy).to(device)
        _, (gold_hist, margin_hist) = train_head_dynamics(
            X, y, num_labels, device, seed=args.seed, record=True)

    # calibration mask for the two conformal scorers: half the documents
    rng2 = np.random.RandomState(args.seed)
    doc_pick = rng2.rand(len(noisy)) < 0.5
    cal_mask = np.zeros(len(flat_noisy), bool)
    for i, take in enumerate(doc_pick):
        if take:
            cal_mask[starts[i]:starts[i + 1]] = True

    scores = {
        'max_softmax': 1.0 - oof[np.arange(len(flat_noisy)), flat_noisy],
        'icp_pvalue': score_icp_pvalue(oof, flat_noisy, cal_mask),
        'mondrian_gap': score_mondrian_gap(oof, flat_noisy, cal_mask),
        'aum': -margin_hist.mean(axis=0),
        'datamap_conf': -gold_hist.mean(axis=0),
        'datamap_var': -gold_hist.std(axis=0),
    }
    try:
        scores['cleanlab'] = score_cleanlab(oof, noisy, starts)
    except Exception as e:                                  # noqa: BLE001
        logger.warning("cleanlab failed: %s", e)
        scores['cleanlab'] = None

    results = {}
    for name, sc in scores.items():
        if sc is None:
            results[name] = {'error': 'unavailable'}
            continue
        r = prf_auprc(sc, flat_flipped)
        r['class_level'] = class_level_ranking(sc, flat_flipped, flat_noisy, id2label)
        results[name] = r

    out = {
        'dataset': args.dataset,
        'trunk': trunk_kind,
        'noise_model': args.noise_model, 'eps': args.eps, 'seed': args.seed,
        'target_class': id2label[target] if target is not None else None,
        'n_tokens': int(len(flat_flipped)),
        'n_planted_errors': int(flat_flipped.sum()),
        'oof_accuracy_vs_noisy_labels': acc,
        'protocol': {
            'oof_folds': K_FOLDS,
            'oof_split_unit': 'document',
            'head': (f'last-{args.finetune_layers}-layer fine-tune on NOISY '
                     f'labels, {args.ft_epochs} epochs, lr {args.ft_lr}'
                     if args.finetune_layers > 0
                     else f'linear probe, {HEAD_EPOCHS} epochs, lr {HEAD_LR}'),
            'note': ('All detectors consume the SAME out-of-fold probabilities, '
                     'except AUM and Data Maps which by definition use training '
                     'dynamics from a single full-data fit. mondrian_gap is '
                     'class-granular by construction: every token of a class '
                     'receives that class coverage deficit, so it is not '
                     'expected to win a token-level ranking metric -- the '
                     'comparison is about granularity, not ranking quality.'),
        },
        'detectors': results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f'\n{"detector":<16}{"AUPRC":>9}{"lift":>8}{"P@n":>9}{"R@n":>9}')
    for name, r in results.items():
        if 'auprc' not in r or r['auprc'] is None:
            print(f'{name:<16}{"n/a":>9}')
            continue
        k = next(iter(r['at_k']))
        cl = r.get('class_level', {})
        rk = cl.get('rank_of_corrupted_class')
        print(f'{name:<16}{r["auprc"]:9.4f}{r["lift_over_random"]:8.1f}'
              f'{r["at_k"][k]["precision"]:9.3f}{r["at_k"][k]["recall"]:9.3f}'
              f'{("  class-rank " + str(rk) + "/" + str(cl.get("n_classes"))) if rk else ""}')
    logger.info("Wrote %s", args.output)


if __name__ == '__main__':
    main()
