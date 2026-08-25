#!/usr/bin/env python3
"""
Generate Qualitative Prediction Visualizations for CaliDoc Paper

Loads trained FUNSD and CORD models, runs inference on test sets, and
produces overlay figures showing:
  - Token-level entity labels with colored bounding boxes
  - Prediction set sizes annotated per token
  - Selected panels: success, challenging, failure caught by CP, typical

Usage:
    python scripts/generate_qualitative_predictions.py \
        --funsd_checkpoint outputs/funsd_aug_6631427/best \
        --cord_checkpoint outputs/cord_optimized_6530869/best \
        --data_root data \
        --output_dir docs/paper/figures
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from calidoc.models.conformal_calibrator import ConformalCalibrator
    HAS_CALIDOC = True
except ImportError:
    HAS_CALIDOC = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Color palette for entity classes
ENTITY_COLORS = {
    # FUNSD
    'O': '#CCCCCC',
    'B-HEADER': '#E74C3C', 'I-HEADER': '#E74C3C',
    'B-QUESTION': '#3498DB', 'I-QUESTION': '#3498DB',
    'B-ANSWER': '#2ECC71', 'I-ANSWER': '#2ECC71',
    # CORD (selected)
    'B-MENU.NM': '#E74C3C', 'I-MENU.NM': '#E74C3C',
    'B-MENU.PRICE': '#3498DB', 'I-MENU.PRICE': '#3498DB',
    'B-TOTAL.TOTAL_PRICE': '#2ECC71', 'I-TOTAL.TOTAL_PRICE': '#2ECC71',
    'B-MENU.CNT': '#9B59B6', 'I-MENU.CNT': '#9B59B6',
    'B-SUB_TOTAL.SUBTOTAL_PRICE': '#F39C12', 'I-SUB_TOTAL.SUBTOTAL_PRICE': '#F39C12',
}
DEFAULT_COLOR = '#95A5A6'


def load_model_and_data(checkpoint_path, dataset_name, data_root,
                        pretrained_name='microsoft/layoutlmv3-large'):
    """Load model and dataset (test split)."""
    from transformers import AutoProcessor
    from calidoc.models import CaliDocModel

    num_labels = 7 if dataset_name == 'funsd' else 53
    training_config_path = os.path.join(checkpoint_path, 'training_config.json')
    if os.path.exists(training_config_path):
        with open(training_config_path) as f:
            tc = json.load(f)
        num_labels = tc.get('num_labels', num_labels)

    model = CaliDocModel.from_pretrained(
        checkpoint_path,
        encoder_config={'model_name': pretrained_name, 'task': 'ner',
                        'num_labels': num_labels},
        use_quality_routing=False, use_calibration=False)
    processor = AutoProcessor.from_pretrained(pretrained_name, apply_ocr=False)

    # Check if local Arrow data exists; if not, use HF cache (offline mode)
    def _has_arrow(dname, split):
        p = os.path.join(data_root, dname, split)
        return os.path.isdir(p) and os.path.isfile(os.path.join(p, 'dataset_info.json'))

    if dataset_name == 'funsd':
        from calidoc.data.funsd_dataset import FUNSDDataset
        use_local = _has_arrow('funsd', 'test')
        logger.info(f"FUNSD: local Arrow={'found' if use_local else 'NOT found, using HF cache'}")
        dataset = FUNSDDataset(
            data_dir=os.path.join(data_root, 'funsd'),
            processor=processor, split='test', max_length=512,
            use_huggingface=not use_local)
    else:
        from calidoc.data.cord_dataset import CORDDataset
        use_local = _has_arrow('cord', 'test')
        logger.info(f"CORD: local Arrow={'found' if use_local else 'NOT found, using HF cache'}")
        dataset = CORDDataset(
            data_dir=os.path.join(data_root, 'cord'),
            processor=processor, split='test', max_length=512,
            use_huggingface=not use_local)
    logger.info(f"Loaded {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=getattr(dataset, 'collate_fn', None))
    id2label = getattr(dataset, 'id2label', None)
    return model, processor, loader, dataset, id2label, num_labels


def run_inference_on_dataset(model, loader, device):
    """Run inference and collect per-document predictions."""
    model.eval().to(device)
    documents = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
            outputs = model(
                images=batch_dev.get('pixel_values'),
                input_ids=batch_dev.get('input_ids'),
                attention_mask=batch_dev.get('attention_mask'),
                bbox=batch_dev.get('bbox'),
                labels=batch_dev.get('labels'),
                return_loss=False)

            logits = outputs['logits'].cpu()  # [1, seq_len, C]
            labels = batch['labels']  # [1, seq_len]
            mask = batch['attention_mask']  # [1, seq_len]
            bbox = batch.get('bbox', None)  # [1, seq_len, 4]
            layout_attn = outputs.get('layout_attention')
            if layout_attn is not None:
                layout_attn = layout_attn.cpu()

            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            confs = probs.max(dim=-1).values

            documents.append({
                'doc_idx': batch_idx,
                'logits': logits.squeeze(0),
                'labels': labels.squeeze(0),
                'mask': mask.squeeze(0),
                'preds': preds.squeeze(0),
                'confs': confs.squeeze(0),
                'bbox': bbox.squeeze(0) if bbox is not None else None,
                'layout_attention': layout_attn.squeeze(0) if layout_attn is not None else None,
            })

    return documents


def calibrate_and_predict(documents, id2label, num_classes, device,
                          alpha=0.1):
    """Calibrate on 70% of docs, predict on 30%."""
    n = len(documents)
    if n == 0:
        raise RuntimeError("No documents loaded — check data_root and dataset paths")
    n_cal = int(n * 0.7)
    if n_cal == 0:
        n_cal = max(1, n // 2)  # ensure at least 1 cal doc
    cal_docs = documents[:n_cal]
    test_docs = documents[n_cal:]
    logger.info(f"Split: {n} total docs → {len(cal_docs)} calibration, {len(test_docs)} test")

    # Flatten calibration tokens
    cal_logits_list, cal_labels_list = [], []
    for doc in cal_docs:
        for t in range(len(doc['labels'])):
            if doc['mask'][t] == 0 or doc['labels'][t] < 0:
                continue
            cal_logits_list.append(doc['logits'][t])
            cal_labels_list.append(doc['labels'][t])

    if len(cal_logits_list) == 0:
        raise RuntimeError(
            f"No valid calibration tokens found in {len(cal_docs)} docs. "
            f"All labels may be -100 (padding). Check dataset loading."
        )
    logger.info(f"Calibration: {len(cal_logits_list)} valid tokens")
    cal_logits = torch.stack(cal_logits_list)
    cal_labels = torch.stack(cal_labels_list)

    calibrator = ConformalCalibrator(
        alpha=alpha, score_type='lac', temperature=1.0).to(device)
    calibrator.calibrate(cal_logits.to(device), cal_labels.to(device),
                         optimize_temperature=True)

    # Predict on test docs
    for doc in test_docs:
        sets_list, sizes_list = [], []
        for t in range(len(doc['labels'])):
            if doc['mask'][t] == 0 or doc['labels'][t] < 0:
                sets_list.append([])
                sizes_list.append(0)
                continue
            tok_logits = doc['logits'][t:t+1].to(device)
            res = calibrator.predict(tok_logits)
            sets_list.append(res[0].prediction_set)
            sizes_list.append(res[0].set_size)
        doc['prediction_sets'] = sets_list
        doc['set_sizes'] = sizes_list

    return test_docs, calibrator


def select_examples(test_docs, id2label):
    """Select representative examples for qualitative visualization."""
    scored = []
    for doc in test_docs:
        valid_confs = []
        valid_sizes = []
        n_correct = 0
        n_total = 0
        has_large_set_on_error = False

        for t in range(len(doc['labels'])):
            if doc['mask'][t] == 0 or doc['labels'][t] < 0:
                continue
            valid_confs.append(doc['confs'][t].item())
            if 'set_sizes' in doc:
                valid_sizes.append(doc['set_sizes'][t])
            n_total += 1
            if doc['preds'][t] == doc['labels'][t]:
                n_correct += 1
            elif 'set_sizes' in doc and doc['set_sizes'][t] > 1:
                has_large_set_on_error = True

        if not valid_confs:
            continue

        scored.append({
            'doc': doc,
            'mean_conf': np.mean(valid_confs),
            'mean_set_size': np.mean(valid_sizes) if valid_sizes else 0,
            'accuracy': n_correct / n_total if n_total > 0 else 0,
            'n_tokens': n_total,
            'has_large_set_on_error': has_large_set_on_error,
        })

    if not scored:
        return {}

    # Sort by different criteria
    by_conf = sorted(scored, key=lambda x: x['mean_conf'], reverse=True)
    by_size = sorted(scored, key=lambda x: x['mean_set_size'])

    selected = {}
    # Success: highest confidence + smallest sets
    selected['success'] = by_conf[0]['doc']
    # Challenging: lowest confidence
    selected['challenging'] = by_conf[-1]['doc']
    # Failure caught by CP: incorrect predictions with large prediction sets
    caught = [s for s in scored if s['has_large_set_on_error']]
    if caught:
        selected['failure_caught'] = caught[0]['doc']
    # Typical: median confidence
    mid = len(by_conf) // 2
    selected['typical'] = by_conf[mid]['doc']

    return selected


def render_document_overlay(doc, id2label, ax, title=''):
    """Render a single document with colored entity boxes and set sizes."""
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)  # inverted y
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_facecolor('#F8F8F8')

    for t in range(len(doc['labels'])):
        if doc['mask'][t] == 0 or doc['labels'][t] < 0:
            continue
        if doc['bbox'] is None:
            continue

        bbox = doc['bbox'][t].tolist()
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            continue

        true_label = doc['labels'][t].item()
        pred_label = doc['preds'][t].item()
        label_name = id2label.get(true_label, str(true_label)) if id2label else str(true_label)

        color = ENTITY_COLORS.get(label_name, DEFAULT_COLOR)
        is_correct = (true_label == pred_label)

        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.5 if not is_correct else 1.0,
            edgecolor='red' if not is_correct else color,
            facecolor=color, alpha=0.35)
        ax.add_patch(rect)

        # Annotate set size if available
        if 'set_sizes' in doc and doc['set_sizes'][t] > 0:
            ss = doc['set_sizes'][t]
            if ss > 1:
                ax.text(x1 + 3, y0 + (y1 - y0) / 2, f'{ss}',
                        fontsize=8, color='#333', va='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='yellow', edgecolor='#999',
                                  alpha=0.85))

    ax.set_xticks([])
    ax.set_yticks([])


def generate_qualitative_figure(selected, id2label, output_path):
    """Generate a multi-panel figure with selected examples."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping figure generation")
        return

    panels = [(k, v) for k, v in selected.items() if v is not None]
    n_panels = len(panels)
    if n_panels == 0:
        logger.warning("No examples selected for visualization")
        return

    if n_panels == 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

    panel_titles = {
        'success': 'High Confidence\n(Smallest Sets)',
        'challenging': 'Low Confidence\n(Challenging)',
        'failure_caught': 'Error Caught by CP\n(Large Pred. Set)',
        'typical': 'Typical Example\n(Median Confidence)',
    }

    for ax, (key, doc) in zip(axes, panels):
        render_document_overlay(doc, id2label, ax,
                                title=panel_titles.get(key, key))

    # Add legend
    legend_elements = []
    seen = set()
    for name, color in ENTITY_COLORS.items():
        if name.startswith('B-') and color not in seen:
            legend_elements.append(
                patches.Patch(facecolor=color, alpha=0.4,
                              label=name.replace('B-', '')))
            seen.add(color)
    legend_elements.append(Line2D([0], [0], color='red', lw=1.5,
                                  label='Incorrect prediction'))

    fig.legend(handles=legend_elements, loc='lower center',
               ncol=min(len(legend_elements), 6), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved qualitative figure to {output_path}")


def save_predictions_json(test_docs, id2label, output_path):
    """Save per-document prediction data to JSON."""
    records = []
    for doc in test_docs:
        tokens = []
        for t in range(len(doc['labels'])):
            if doc['mask'][t] == 0 or doc['labels'][t] < 0:
                continue
            tl = doc['labels'][t].item()
            pl = doc['preds'][t].item()
            entry = {
                'true_label': id2label.get(tl, str(tl)) if id2label else str(tl),
                'pred_label': id2label.get(pl, str(pl)) if id2label else str(pl),
                'confidence': round(doc['confs'][t].item(), 6),
            }
            if doc['bbox'] is not None:
                entry['bbox'] = doc['bbox'][t].tolist()
            if 'prediction_sets' in doc:
                entry['prediction_set'] = doc['prediction_sets'][t]
                entry['set_size'] = doc['set_sizes'][t]
            tokens.append(entry)

        records.append({
            'doc_id': doc['doc_idx'],
            'n_tokens': len(tokens),
            'tokens': tokens,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved predictions JSON ({len(records)} docs) to {output_path}")


def render_document_from_json(tokens, ax, title=''):
    """Render a document from JSON token list (no model needed)."""
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_facecolor('#F8F8F8')

    for tok in tokens:
        if 'bbox' not in tok:
            continue
        x0, y0, x1, y1 = tok['bbox']
        if x1 <= x0 or y1 <= y0:
            continue

        label_name = tok['true_label']
        color = ENTITY_COLORS.get(label_name, DEFAULT_COLOR)
        is_correct = (tok['pred_label'] == tok['true_label'])

        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.5 if not is_correct else 1.0,
            edgecolor='red' if not is_correct else color,
            facecolor=color, alpha=0.35)
        ax.add_patch(rect)

        ss = tok.get('set_size', 0)
        if ss > 1:
            ax.text(x1 + 3, y0 + (y1 - y0) / 2, f'{ss}',
                    fontsize=8, color='#333', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='yellow', edgecolor='#999',
                              alpha=0.85))

    ax.set_xticks([])
    ax.set_yticks([])


def regenerate_from_json(json_path, output_dir, dataset_name):
    """Regenerate qualitative figure from saved JSON (no model needed)."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping")
        return

    with open(json_path) as f:
        records = json.load(f)

    # Score each document for selection
    scored = []
    for rec in records:
        tokens = rec['tokens']
        if not tokens:
            continue
        confs = [t['confidence'] for t in tokens]
        sizes = [t.get('set_size', 0) for t in tokens
                 if t.get('set_size', 0) > 0]
        n_correct = sum(1 for t in tokens
                        if t['pred_label'] == t['true_label'])
        has_large_set_on_error = any(
            t.get('set_size', 0) > 1
            and t['pred_label'] != t['true_label']
            for t in tokens)
        scored.append({
            'tokens': tokens,
            'mean_conf': np.mean(confs),
            'mean_set_size': np.mean(sizes) if sizes else 0,
            'has_large_set_on_error': has_large_set_on_error,
        })

    if not scored:
        logger.warning(f"No documents in {json_path}")
        return

    by_conf = sorted(scored, key=lambda x: x['mean_conf'], reverse=True)
    selected = {}
    selected['success'] = by_conf[0]['tokens']
    selected['challenging'] = by_conf[-1]['tokens']
    caught = [s for s in scored if s['has_large_set_on_error']]
    if caught:
        selected['failure_caught'] = caught[0]['tokens']
    mid = len(by_conf) // 2
    selected['typical'] = by_conf[mid]['tokens']

    panels = [(k, v) for k, v in selected.items() if v is not None]
    n_panels = len(panels)
    if n_panels == 0:
        logger.warning("No examples selected")
        return

    if n_panels == 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

    panel_titles = {
        'success': 'High Confidence\n(Smallest Sets)',
        'challenging': 'Low Confidence\n(Challenging)',
        'failure_caught': 'Error Caught by CP\n(Large Pred. Set)',
        'typical': 'Typical Example\n(Median Confidence)',
    }

    for ax, (key, tokens) in zip(axes, panels):
        render_document_from_json(tokens, ax,
                                  title=panel_titles.get(key, key))

    # Legend
    legend_elements = []
    seen = set()
    for name, color in ENTITY_COLORS.items():
        if name.startswith('B-') and color not in seen:
            legend_elements.append(
                patches.Patch(facecolor=color, alpha=0.4,
                              label=name.replace('B-', '')))
            seen.add(color)
    legend_elements.append(Line2D([0], [0], color='red', lw=2,
                                  label='Incorrect prediction'))

    fig.legend(handles=legend_elements, loc='lower center',
               ncol=min(len(legend_elements), 6), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(
        output_dir, f'qualitative_predictions_{dataset_name}.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved regenerated figure to {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative prediction visualizations")
    parser.add_argument('--funsd_checkpoint', type=str, default=None)
    parser.add_argument('--cord_checkpoint', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str,
                        default='docs/paper/figures')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrained_name', type=str,
                        default='microsoft/layoutlmv3-large')
    parser.add_argument('--from_json', action='store_true',
                        help='Regenerate from saved JSON (no model needed)')
    parser.add_argument('--funsd_json', type=str, default=None,
                        help='Path to FUNSD predictions JSON')
    parser.add_argument('--cord_json', type=str, default=None,
                        help='Path to CORD predictions JSON')
    args = parser.parse_args()

    # JSON-only mode: regenerate from saved predictions
    if args.from_json:
        for dataset_name, json_path in [('funsd', args.funsd_json),
                                         ('cord', args.cord_json)]:
            if json_path and os.path.exists(json_path):
                regenerate_from_json(json_path, args.output_dir,
                                     dataset_name)
            else:
                logger.info(f"Skipping {dataset_name} (no JSON path)")
        logger.info("\nDone.")
        return

    if not HAS_CALIDOC:
        logger.error("calidoc package not found. Use --from_json mode.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    for dataset_name, ckpt in [('funsd', args.funsd_checkpoint),
                                ('cord', args.cord_checkpoint)]:
        if ckpt is None:
            logger.info(f"Skipping {dataset_name} (no checkpoint provided)")
            continue

        logger.info(f"\n{'='*60}\nProcessing {dataset_name}\n{'='*60}")

        model, processor, loader, dataset, id2label, num_labels = \
            load_model_and_data(ckpt, dataset_name, args.data_root,
                                args.pretrained_name)

        documents = run_inference_on_dataset(model, loader, device)

        # Calibrate and get prediction sets
        test_docs, calibrator = calibrate_and_predict(
            documents, id2label, num_labels, device, alpha=0.1)

        # Select examples
        selected = select_examples(test_docs, id2label)
        logger.info(f"Selected panels: {list(selected.keys())}")

        # Generate figure
        fig_path = os.path.join(
            args.output_dir,
            f'qualitative_predictions_{dataset_name}.pdf')
        generate_qualitative_figure(selected, id2label, fig_path)

        # Save predictions JSON
        json_path = os.path.join(
            args.output_dir,
            f'qualitative_predictions_{dataset_name}.json')
        save_predictions_json(test_docs, id2label, json_path)

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
