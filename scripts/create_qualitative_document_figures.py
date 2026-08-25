#!/usr/bin/env python3
"""
Create Document-Level Qualitative Visualizations.

This script generates publication figures showing:
1. Document images with predicted bounding boxes and labels
2. Side-by-side comparison: Ground truth vs Predictions
3. Attention overlaid on document images
4. Uncertainty visualization on documents

These are CRITICAL figures for paper acceptance - reviewers need to see
the model actually working on real documents.
"""

import os
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Entity colors (vibrant for visibility)
ENTITY_COLORS = {
    # FUNSD
    'O': (0.9, 0.9, 0.9, 0.3),
    'B-HEADER': (0.13, 0.59, 0.95, 0.6),   # Blue
    'I-HEADER': (0.13, 0.59, 0.95, 0.4),
    'B-QUESTION': (0.30, 0.69, 0.31, 0.6),  # Green
    'I-QUESTION': (0.30, 0.69, 0.31, 0.4),
    'B-ANSWER': (1.0, 0.60, 0.0, 0.6),      # Orange
    'I-ANSWER': (1.0, 0.60, 0.0, 0.4),
    # Generic
    'HEADER': (0.13, 0.59, 0.95, 0.6),
    'QUESTION': (0.30, 0.69, 0.31, 0.6),
    'ANSWER': (1.0, 0.60, 0.0, 0.6),
    'OTHER': (0.5, 0.5, 0.5, 0.3),
}


def normalize_box(box, width, height):
    """Normalize box coordinates to 0-1000 range."""
    return [
        int(box[0] * 1000 / width),
        int(box[1] * 1000 / height),
        int(box[2] * 1000 / width),
        int(box[3] * 1000 / height),
    ]


def denormalize_box(box, width, height):
    """Denormalize box from 0-1000 to pixel coordinates."""
    return [
        int(box[0] * width / 1000),
        int(box[1] * height / 1000),
        int(box[2] * width / 1000),
        int(box[3] * height / 1000),
    ]


def get_label_name(label_id, label_map):
    """Convert label ID to name."""
    if isinstance(label_map, dict):
        return label_map.get(label_id, f'LABEL_{label_id}')
    return f'LABEL_{label_id}'


def draw_boxes_on_image(image, words, boxes, labels, label_map, confidences=None):
    """Draw labeled bounding boxes on document image."""
    img = image.copy()
    draw = ImageDraw.Draw(img, 'RGBA')

    width, height = img.size

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()

    for i, (word, box, label) in enumerate(zip(words, boxes, labels)):
        if label == -100:
            continue

        label_name = get_label_name(label, label_map)

        # Skip 'O' labels for cleaner visualization
        if label_name == 'O' or label_name.startswith('LABEL_0'):
            continue

        # Get color
        color = ENTITY_COLORS.get(label_name, ENTITY_COLORS.get('OTHER'))

        # Denormalize box if needed
        if max(box) <= 1000:
            box = denormalize_box(box, width, height)

        x1, y1, x2, y2 = box

        # Draw filled rectangle
        draw.rectangle([x1, y1, x2, y2], fill=tuple(int(c * 255) for c in color))

        # Draw border
        border_color = tuple(int(c * 255) for c in color[:3]) + (255,)
        draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)

        # Add confidence if available
        if confidences is not None and i < len(confidences):
            conf = confidences[i]
            if conf < 0.9:  # Only show confidence for uncertain predictions
                draw.text((x1, y1 - 12), f'{conf:.0%}', fill=(255, 0, 0), font=font)

    return img


def create_comparison_figure(image, words, boxes, gt_labels, pred_labels,
                             label_map, confidences, output_path, title=""):
    """Create side-by-side comparison of ground truth vs predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Document', fontsize=14)
    axes[0].axis('off')

    # Ground truth
    gt_img = draw_boxes_on_image(image, words, boxes, gt_labels, label_map)
    axes[1].imshow(gt_img)
    axes[1].set_title('Ground Truth Labels', fontsize=14)
    axes[1].axis('off')

    # Predictions
    pred_img = draw_boxes_on_image(image, words, boxes, pred_labels, label_map, confidences)
    axes[2].imshow(pred_img)
    axes[2].set_title('Model Predictions', fontsize=14)
    axes[2].axis('off')

    # Add legend
    legend_elements = []
    for label_name, color in ENTITY_COLORS.items():
        if label_name not in ['O', 'OTHER'] and not label_name.startswith('I-'):
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='black',
                              label=label_name.replace('B-', ''))
            )

    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.02))

    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{output_path}.{fmt}")
    plt.close()


def create_error_highlight_figure(image, words, boxes, gt_labels, pred_labels,
                                  label_map, confidences, output_path):
    """Create figure highlighting errors with detailed annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Draw base image
    ax.imshow(image)

    width, height = image.size

    errors = []
    correct = []

    for i, (word, box, gt, pred) in enumerate(zip(words, boxes, gt_labels, pred_labels)):
        if gt == -100:
            continue

        if max(box) <= 1000:
            box = denormalize_box(box, width, height)

        x1, y1, x2, y2 = box

        if gt != pred:
            # Error - highlight in red
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='red', facecolor='red', alpha=0.3
            )
            ax.add_patch(rect)
            errors.append({
                'word': word, 'box': box,
                'gt': get_label_name(gt, label_map),
                'pred': get_label_name(pred, label_map),
                'conf': confidences[i] if i < len(confidences) else 0
            })
        else:
            gt_name = get_label_name(gt, label_map)
            if gt_name != 'O':
                # Correct entity - highlight in green
                color = ENTITY_COLORS.get(gt_name, (0.3, 0.7, 0.3, 0.3))
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='green', facecolor=color, alpha=0.4
                )
                ax.add_patch(rect)
                correct.append({'word': word, 'label': gt_name})

    ax.set_title(f'Error Analysis: {len(errors)} errors, {len(correct)} correct entities', fontsize=14)
    ax.axis('off')

    # Add error annotations on the side
    error_text = "ERRORS:\n" + "\n".join([
        f"'{e['word'][:15]}': {e['gt']}→{e['pred']} ({e['conf']:.0%})"
        for e in errors[:10]
    ])

    ax.text(1.02, 0.95, error_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{output_path}.{fmt}")
    plt.close()


def create_confidence_overlay(image, words, boxes, confidences, output_path):
    """Create heatmap overlay showing prediction confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original Document', fontsize=14)
    axes[0].axis('off')

    # Confidence overlay
    axes[1].imshow(image)

    width, height = image.size

    for word, box, conf in zip(words, boxes, confidences):
        if max(box) <= 1000:
            box = denormalize_box(box, width, height)

        x1, y1, x2, y2 = box

        # Color based on confidence: green (high) to red (low)
        if conf >= 0.9:
            color = (0.3, 0.8, 0.3, 0.4)  # Green
        elif conf >= 0.7:
            color = (1.0, 0.8, 0.0, 0.4)  # Yellow
        else:
            color = (1.0, 0.3, 0.3, 0.5)  # Red

        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1, edgecolor='black', facecolor=color
        )
        axes[1].add_patch(rect)

    axes[1].set_title('Prediction Confidence Overlay', fontsize=14)
    axes[1].axis('off')

    # Add colorbar legend
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('conf', ['red', 'yellow', 'green'])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.5, 1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Confidence', fontsize=10)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{output_path}.{fmt}")
    plt.close()


def create_attention_overlay(image, attention_weights, words, boxes, output_path):
    """Create attention heatmap overlaid on document."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    width, height = image.size

    # Average attention over heads
    avg_attention = attention_weights.mean(axis=0)  # (seq_len, seq_len)

    # Get attention to [CLS] token (global document representation)
    cls_attention = avg_attention[0, :]

    # Get self-attention (diagonal)
    self_attention = np.diag(avg_attention)

    # Normalize
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
    self_attention = (self_attention - self_attention.min()) / (self_attention.max() - self_attention.min() + 1e-8)

    for ax_idx, (attention, title) in enumerate([
        (None, 'Original Document'),
        (cls_attention, 'Global Attention ([CLS])'),
        (self_attention, 'Self-Attention (Diagonal)'),
    ]):
        axes[ax_idx].imshow(image)

        if attention is not None:
            for i, (word, box) in enumerate(zip(words, boxes)):
                if i >= len(attention):
                    break

                if max(box) <= 1000:
                    box = denormalize_box(box, width, height)

                x1, y1, x2, y2 = box
                attn_val = attention[i]

                # Color intensity based on attention
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=1, edgecolor='blue',
                    facecolor='blue', alpha=attn_val * 0.7
                )
                axes[ax_idx].add_patch(rect)

        axes[ax_idx].set_title(title, fontsize=14)
        axes[ax_idx].axis('off')

    plt.suptitle('Attention Pattern Visualization', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{output_path}.{fmt}")
    plt.close()


def create_multi_sample_grid(samples, output_path, task):
    """Create grid showing multiple document samples."""
    n_samples = min(6, len(samples))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, sample in enumerate(samples[:n_samples]):
        ax = axes[idx // 3, idx % 3]

        # Load or create image
        if 'image' in sample and sample['image'] is not None:
            image = sample['image']
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
        else:
            # Create placeholder
            image = Image.new('RGB', (800, 1000), color='white')

        # Draw predictions
        words = sample.get('words', [])
        boxes = sample.get('boxes', [])
        preds = sample.get('predictions', [])
        confs = sample.get('confidences', [])
        label_map = sample.get('label_map', {})

        img_with_boxes = draw_boxes_on_image(image, words, boxes, preds, label_map, confs)

        ax.imshow(img_with_boxes)
        accuracy = sample.get('accuracy', 0)
        ax.set_title(f"Sample {sample.get('sample_idx', idx)}\nAccuracy: {accuracy:.1%}", fontsize=11)
        ax.axis('off')

    plt.suptitle(f'{task.upper()}: Model Predictions on Test Documents', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{output_path}.{fmt}")
    plt.close()
    print(f"Created: {output_path}.pdf")


def create_publication_qualitative_figure(samples, output_path, task):
    """Create the main publication qualitative figure."""
    # Select 3 samples: best, medium, worst
    samples_sorted = sorted(samples, key=lambda x: x.get('accuracy', 0))

    worst = samples_sorted[0]
    medium = samples_sorted[len(samples_sorted)//2]
    best = samples_sorted[-1]

    selected = [best, medium, worst]
    titles = ['High Accuracy Example', 'Medium Accuracy Example', 'Challenging Example']

    fig = plt.figure(figsize=(16, 14))

    for idx, (sample, title) in enumerate(zip(selected, titles)):
        # Create subplot for this sample
        ax = fig.add_subplot(3, 2, idx*2 + 1)

        # Create placeholder image if needed
        if 'image' in sample and sample['image'] is not None:
            image = sample['image']
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
        else:
            image = Image.new('RGB', (600, 800), color='white')

        words = sample.get('words', [])[:50]
        boxes = sample.get('boxes', [])[:50]
        gt_labels = sample.get('labels', [])[:50]
        pred_labels = sample.get('predictions', [])[:50]
        confs = sample.get('confidences', [])[:50]
        label_map = sample.get('label_map', {})

        # Ground truth
        gt_img = draw_boxes_on_image(image, words, boxes, gt_labels, label_map)
        ax.imshow(gt_img)
        ax.set_title(f'{title}\nGround Truth', fontsize=11)
        ax.axis('off')

        # Predictions
        ax2 = fig.add_subplot(3, 2, idx*2 + 2)
        pred_img = draw_boxes_on_image(image, words, boxes, pred_labels, label_map, confs)
        ax2.imshow(pred_img)

        accuracy = sample.get('accuracy', 0)
        ax2.set_title(f'Predictions (Acc: {accuracy:.1%})', fontsize=11)
        ax2.axis('off')

    # Legend
    legend_elements = [
        patches.Patch(facecolor=ENTITY_COLORS['HEADER'], edgecolor='black', label='HEADER'),
        patches.Patch(facecolor=ENTITY_COLORS['QUESTION'], edgecolor='black', label='QUESTION'),
        patches.Patch(facecolor=ENTITY_COLORS['ANSWER'], edgecolor='black', label='ANSWER'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(f'{task.upper()}: Qualitative Results - Ground Truth vs Predictions',
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    for fmt in ['pdf', 'png']:
        plt.savefig(f"{output_path}.{fmt}")
    plt.close()
    print(f"Created: {output_path}.pdf")


def main():
    parser = argparse.ArgumentParser(description="Create document-level qualitative figures")
    parser.add_argument('--analysis_dir', type=str, default='./analysis_outputs')
    parser.add_argument('--output_dir', type=str, default='./docs/figures')
    parser.add_argument('--task', type=str, default='funsd', choices=['funsd', 'cord'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("Creating Document-Level Qualitative Figures")
    print("="*60)

    # Load qualitative data
    qual_path = f"{args.analysis_dir}/{args.task}_qualitative.json"

    if os.path.exists(qual_path):
        print(f"Loading qualitative data from {qual_path}")
        with open(qual_path, 'r') as f:
            samples = json.load(f)

        # Create main publication figure
        create_publication_qualitative_figure(
            samples,
            f"{args.output_dir}/qualitative_main_{args.task}",
            args.task
        )

        # Create multi-sample grid
        create_multi_sample_grid(
            samples,
            f"{args.output_dir}/qualitative_grid_{args.task}",
            args.task
        )

    else:
        print(f"No qualitative data found at {qual_path}")
        print("Please run extract_model_analysis.py on the cluster first")
        print("\nCreating placeholder figures with synthetic data...")

        # Create synthetic samples for demonstration
        synthetic_samples = []
        for i in range(6):
            synthetic_samples.append({
                'sample_idx': i,
                'words': [f'word{j}' for j in range(30)],
                'boxes': [[j*30, 100, j*30+25, 120] for j in range(30)],
                'labels': [j % 7 for j in range(30)],
                'predictions': [j % 7 if np.random.random() > 0.1 else (j+1) % 7 for j in range(30)],
                'confidences': [0.7 + np.random.random() * 0.3 for _ in range(30)],
                'accuracy': 0.85 + np.random.random() * 0.15 - i * 0.1,
                'label_map': {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION',
                              4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'},
                'image': None,
            })

        create_publication_qualitative_figure(
            synthetic_samples,
            f"{args.output_dir}/qualitative_main_{args.task}_placeholder",
            args.task
        )

    # Load attention data if available
    attn_path = f"{args.analysis_dir}/{args.task}_attention_data.pkl"
    if os.path.exists(attn_path):
        print(f"\nLoading attention data from {attn_path}")
        with open(attn_path, 'rb') as f:
            attention_data = pickle.load(f)

        # Create attention visualization for first sample
        if attention_data:
            sample = attention_data[0]
            # Would need actual image to create overlay
            print("Attention data loaded - would create overlay with actual images")

    print(f"\n{'='*60}")
    print(f"Figures saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
