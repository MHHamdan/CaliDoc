#!/usr/bin/env python3
"""
Extract Real Model Outputs for Paper Figures.

This script runs on the cluster to extract:
1. Attention patterns from LayoutLMv3
2. Token embeddings for t-SNE visualization
3. Predictions with confidence scores
4. Qualitative samples for comparison figures

Run on nibi cluster where models are saved.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_layoutlmv3_model(checkpoint_path, device='cuda'):
    """Load trained LayoutLMv3 model with attention output."""
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        checkpoint_path,
        output_attentions=True,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()

    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-large",
        apply_ocr=False
    )

    return model, processor


def extract_attention_patterns(model, processor, dataset, num_samples=10, device='cuda'):
    """Extract attention patterns from specific document samples."""
    attention_data = []

    for idx in tqdm(range(min(num_samples, len(dataset))), desc="Extracting attention"):
        sample = dataset[idx]

        # Prepare inputs
        encoding = processor(
            sample['image'],
            sample['words'],
            boxes=sample['boxes'],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        # Get attention weights from last layer
        # Shape: (batch, num_heads, seq_len, seq_len)
        attentions = outputs.attentions[-1][0].cpu().numpy()  # Last layer

        # Get hidden states for embeddings
        hidden_states = outputs.hidden_states[-1][0].cpu().numpy()  # Last layer

        # Get predictions and confidences
        logits = outputs.logits[0].cpu()
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1).numpy()
        confidences = torch.max(probs, dim=-1).values.numpy()

        attention_data.append({
            'sample_idx': idx,
            'words': sample['words'][:512],
            'boxes': sample['boxes'][:512],
            'labels': sample.get('labels', [])[:512],
            'attentions': attentions,  # (num_heads, seq_len, seq_len)
            'hidden_states': hidden_states,  # (seq_len, hidden_dim)
            'predictions': predictions,
            'confidences': confidences,
            'image_path': sample.get('image_path', ''),
        })

    return attention_data


def extract_embeddings_for_tsne(model, processor, dataset, device='cuda'):
    """Extract embeddings for all samples for t-SNE visualization."""
    all_embeddings = []
    all_labels = []
    all_predictions = []
    all_confidences = []
    all_words = []

    for idx in tqdm(range(len(dataset)), desc="Extracting embeddings"):
        sample = dataset[idx]

        encoding = processor(
            sample['image'],
            sample['words'],
            boxes=sample['boxes'],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**encoding, output_hidden_states=True)

        # Get embeddings from last hidden state
        hidden_states = outputs.hidden_states[-1][0].cpu().numpy()

        # Get predictions
        logits = outputs.logits[0].cpu()
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1).numpy()
        confidences = torch.max(probs, dim=-1).values.numpy()

        # Get actual sequence length (non-padding)
        attention_mask = encoding['attention_mask'][0].cpu().numpy()
        seq_len = int(attention_mask.sum())

        labels = sample.get('labels', [-100] * seq_len)[:seq_len]

        # Store token-level data (skip special tokens)
        for i in range(1, min(seq_len - 1, len(labels))):
            if labels[i] != -100:  # Skip padding/special tokens
                all_embeddings.append(hidden_states[i])
                all_labels.append(labels[i])
                all_predictions.append(predictions[i])
                all_confidences.append(confidences[i])
                if i < len(sample['words']):
                    all_words.append(sample['words'][i])
                else:
                    all_words.append('')

    return {
        'embeddings': np.array(all_embeddings),
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'confidences': np.array(all_confidences),
        'words': all_words,
    }


def extract_qualitative_samples(model, processor, dataset, label_map,
                                num_samples=20, device='cuda'):
    """Extract samples for qualitative comparison figures."""
    samples = []

    # Select diverse samples (some correct, some with errors)
    indices = list(range(min(num_samples * 2, len(dataset))))

    for idx in tqdm(indices, desc="Extracting qualitative samples"):
        sample = dataset[idx]

        encoding = processor(
            sample['image'],
            sample['words'],
            boxes=sample['boxes'],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits[0].cpu()
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1).numpy()
        confidences = torch.max(probs, dim=-1).values.numpy()

        # Get top-k predictions for conformal analysis
        top_k_probs, top_k_indices = torch.topk(probs, k=3, dim=-1)

        labels = sample.get('labels', [])

        # Calculate accuracy for this sample
        correct = 0
        total = 0
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if label != -100:
                total += 1
                if pred == label:
                    correct += 1

        accuracy = correct / total if total > 0 else 0

        samples.append({
            'sample_idx': idx,
            'image_path': sample.get('image_path', ''),
            'words': sample['words'],
            'boxes': sample['boxes'],
            'labels': labels,
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'top_k_probs': top_k_probs.numpy().tolist(),
            'top_k_indices': top_k_indices.numpy().tolist(),
            'accuracy': accuracy,
            'label_map': label_map,
        })

    # Sort by accuracy to get mix of good and bad samples
    samples.sort(key=lambda x: x['accuracy'])

    # Select: 5 worst, 5 medium, 10 best
    selected = samples[:5] + samples[len(samples)//2-2:len(samples)//2+3] + samples[-10:]

    return selected


def extract_per_class_metrics(model, processor, dataset, label_map, device='cuda'):
    """Extract detailed per-class metrics."""
    from collections import defaultdict

    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_confidences_correct = defaultdict(list)
    class_confidences_incorrect = defaultdict(list)

    for idx in tqdm(range(len(dataset)), desc="Computing per-class metrics"):
        sample = dataset[idx]

        encoding = processor(
            sample['image'],
            sample['words'],
            boxes=sample['boxes'],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits[0].cpu()
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1).numpy()
        confidences = torch.max(probs, dim=-1).values.numpy()

        labels = sample.get('labels', [])

        for i, (pred, label, conf) in enumerate(zip(predictions, labels, confidences)):
            if label == -100:
                continue

            if pred == label:
                class_tp[label] += 1
                class_confidences_correct[label].append(conf)
            else:
                class_fp[pred] += 1
                class_fn[label] += 1
                class_confidences_incorrect[label].append(conf)

    # Compute metrics per class
    results = {}
    for class_id, class_name in label_map.items():
        tp = class_tp[class_id]
        fp = class_fp[class_id]
        fn = class_fn[class_id]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        correct_confs = class_confidences_correct[class_id]
        incorrect_confs = class_confidences_incorrect[class_id]

        results[class_name] = {
            'class_id': class_id,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn,
            'avg_confidence_correct': np.mean(correct_confs) if correct_confs else 0,
            'avg_confidence_incorrect': np.mean(incorrect_confs) if incorrect_confs else 0,
            'confidence_correct': correct_confs,
            'confidence_incorrect': incorrect_confs,
        }

    return results


def extract_conformal_analysis(model, processor, dataset, alpha_values=[0.05, 0.1, 0.15, 0.2],
                               device='cuda'):
    """Extract data for conformal prediction sensitivity analysis."""
    all_probs = []
    all_labels = []

    for idx in tqdm(range(len(dataset)), desc="Extracting conformal data"):
        sample = dataset[idx]

        encoding = processor(
            sample['image'],
            sample['words'],
            boxes=sample['boxes'],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits[0].cpu()
        probs = F.softmax(logits, dim=-1).numpy()
        labels = sample.get('labels', [])

        for i, (prob, label) in enumerate(zip(probs, labels)):
            if label != -100:
                all_probs.append(prob)
                all_labels.append(label)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Split into calibration and test
    n = len(all_probs)
    cal_size = n // 2

    cal_probs = all_probs[:cal_size]
    cal_labels = all_labels[:cal_size]
    test_probs = all_probs[cal_size:]
    test_labels = all_labels[cal_size:]

    results = {}

    for alpha in alpha_values:
        # LAC: Compute threshold
        cal_scores = 1 - cal_probs[np.arange(len(cal_labels)), cal_labels]
        threshold = np.quantile(cal_scores, 1 - alpha)

        # Compute prediction sets
        coverages = []
        set_sizes = []

        for prob, label in zip(test_probs, test_labels):
            pred_set = np.where(prob >= (1 - threshold))[0]
            set_sizes.append(len(pred_set))
            coverages.append(1 if label in pred_set else 0)

        results[f'alpha_{alpha}'] = {
            'alpha': alpha,
            'threshold': float(threshold),
            'coverage': float(np.mean(coverages)),
            'avg_set_size': float(np.mean(set_sizes)),
            'set_size_distribution': {
                str(i): int(np.sum(np.array(set_sizes) == i))
                for i in range(1, 6)
            },
        }

    return results


def load_funsd_dataset(data_path):
    """Load FUNSD dataset."""
    from datasets import load_dataset

    dataset = load_dataset("nielsr/funsd-layoutlmv3", split="test")

    # Convert to our format
    samples = []
    for item in dataset:
        samples.append({
            'image': item['image'],
            'words': item['tokens'],
            'boxes': item['bboxes'],
            'labels': item['ner_tags'],
            'image_path': '',
        })

    return samples


def load_cord_dataset(data_path):
    """Load CORD dataset."""
    from datasets import load_dataset

    dataset = load_dataset("naver-clova-ix/cord-v2", split="test")

    samples = []
    for item in dataset:
        # Parse ground truth
        gt = json.loads(item['ground_truth'])
        words = []
        boxes = []
        labels = []

        for line in gt.get('valid_line', []):
            for word_info in line.get('words', []):
                words.append(word_info.get('text', ''))
                quad = word_info.get('quad', {})
                x1 = min(quad.get('x1', 0), quad.get('x4', 0))
                y1 = min(quad.get('y1', 0), quad.get('y2', 0))
                x2 = max(quad.get('x2', 0), quad.get('x3', 0))
                y2 = max(quad.get('y3', 0), quad.get('y4', 0))
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                # Simplified label assignment
                labels.append(0)  # Would need proper label mapping

        samples.append({
            'image': item['image'],
            'words': words,
            'boxes': boxes,
            'labels': labels,
            'image_path': '',
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract model analysis data")
    parser.add_argument('--task', type=str, required=True, choices=['funsd', 'cord'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./analysis_outputs')
    parser.add_argument('--num_attention_samples', type=int, default=10)
    parser.add_argument('--num_qualitative_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, processor = load_layoutlmv3_model(args.checkpoint, args.device)

    # Load dataset
    print(f"Loading {args.task} dataset...")
    if args.task == 'funsd':
        dataset = load_funsd_dataset(None)
        label_map = {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION',
                     4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}
    else:
        dataset = load_cord_dataset(None)
        label_map = {i: f'LABEL_{i}' for i in range(54)}  # CORD has many labels

    print(f"Dataset size: {len(dataset)}")

    # 1. Extract attention patterns
    print("\n1. Extracting attention patterns...")
    attention_data = extract_attention_patterns(
        model, processor, dataset,
        num_samples=args.num_attention_samples,
        device=args.device
    )
    with open(f"{args.output_dir}/{args.task}_attention_data.pkl", 'wb') as f:
        pickle.dump(attention_data, f)
    print(f"   Saved attention data for {len(attention_data)} samples")

    # 2. Extract embeddings for t-SNE
    print("\n2. Extracting embeddings for t-SNE...")
    embedding_data = extract_embeddings_for_tsne(
        model, processor, dataset, device=args.device
    )
    with open(f"{args.output_dir}/{args.task}_embeddings.pkl", 'wb') as f:
        pickle.dump(embedding_data, f)
    print(f"   Saved {len(embedding_data['embeddings'])} token embeddings")

    # 3. Extract qualitative samples
    print("\n3. Extracting qualitative samples...")
    qualitative_samples = extract_qualitative_samples(
        model, processor, dataset, label_map,
        num_samples=args.num_qualitative_samples,
        device=args.device
    )
    with open(f"{args.output_dir}/{args.task}_qualitative.json", 'w') as f:
        json.dump(qualitative_samples, f, indent=2)
    print(f"   Saved {len(qualitative_samples)} qualitative samples")

    # 4. Extract per-class metrics
    print("\n4. Computing per-class metrics...")
    class_metrics = extract_per_class_metrics(
        model, processor, dataset, label_map, device=args.device
    )
    # Remove large confidence lists for JSON
    class_metrics_json = {}
    for k, v in class_metrics.items():
        class_metrics_json[k] = {kk: vv for kk, vv in v.items()
                                  if kk not in ['confidence_correct', 'confidence_incorrect']}
    with open(f"{args.output_dir}/{args.task}_class_metrics.json", 'w') as f:
        json.dump(class_metrics_json, f, indent=2)
    with open(f"{args.output_dir}/{args.task}_class_metrics_full.pkl", 'wb') as f:
        pickle.dump(class_metrics, f)
    print(f"   Saved metrics for {len(class_metrics)} classes")

    # 5. Extract conformal prediction analysis
    print("\n5. Extracting conformal prediction analysis...")
    conformal_data = extract_conformal_analysis(
        model, processor, dataset, device=args.device
    )
    with open(f"{args.output_dir}/{args.task}_conformal.json", 'w') as f:
        json.dump(conformal_data, f, indent=2)
    print(f"   Saved conformal analysis for {len(conformal_data)} alpha values")

    print(f"\n{'='*50}")
    print(f"All analysis data saved to: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
