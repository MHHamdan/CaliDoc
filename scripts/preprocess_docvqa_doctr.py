#!/usr/bin/env python3
"""
Pre-process DocVQA dataset with OCR annotations using docTR.

docTR is a pure PyTorch OCR library that works without opencv module.
It uses deep learning models for text detection and recognition.

Features:
- Checkpointing: Saves progress every batch, resumes from last checkpoint
- Memory efficient: Processes in batches, clears memory periodically
- GPU accelerated: Uses GPU if available

Usage:
    python scripts/preprocess_docvqa_doctr.py --output_dir ./data/docvqa
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
import torch


def setup_doctr():
    """Setup docTR with PyTorch backend."""
    # Force PyTorch backend before importing doctr
    os.environ['USE_TORCH'] = '1'

    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-trained OCR model
    print("Loading docTR OCR model...")
    model = ocr_predictor(
        det_arch='db_resnet50',
        reco_arch='crnn_vgg16_bn',
        pretrained=True
    )

    if torch.cuda.is_available():
        model = model.cuda()

    return model, device


def run_ocr_doctr(image, model):
    """
    Run docTR OCR on an image and return words and boxes.
    """
    from doctr.io import DocumentFile

    if isinstance(image, Image.Image):
        img = image.convert('RGB')
    else:
        img = Image.open(image).convert('RGB')

    width, height = img.size

    # Convert PIL to numpy array
    img_array = np.array(img)

    # Run OCR
    result = model([img_array])

    words = []
    boxes = []

    # Parse results
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text = word.value.strip()
                    if not text:
                        continue

                    # docTR returns normalized coordinates (0-1)
                    # Convert to LayoutLMv3 format (0-1000)
                    (x1, y1), (x2, y2) = word.geometry

                    box = [
                        int(1000 * x1),
                        int(1000 * y1),
                        int(1000 * x2),
                        int(1000 * y2),
                    ]
                    box = [max(0, min(1000, v)) for v in box]

                    words.append(text)
                    boxes.append(box)

    return words, boxes


def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'processed_batches': [], 'last_idx': 0}


def save_checkpoint(checkpoint_path, checkpoint_data):
    """Save checkpoint."""
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)


def process_split(dataset, split_name, output_dir, model, batch_size=500):
    """Process a dataset split with checkpointing."""
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / '_checkpoint.json'
    checkpoint = load_checkpoint(checkpoint_path)

    start_idx = checkpoint['last_idx']
    total = len(dataset)

    if start_idx >= total:
        print(f"Split {split_name} already fully processed ({total} samples)")
        return

    if start_idx > 0:
        print(f"Resuming {split_name} from sample {start_idx}/{total}")
    else:
        print(f"\nProcessing {split_name} split ({total} samples)...")

    batch_num = start_idx // batch_size

    for batch_start in range(start_idx, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_samples = []

        pbar = tqdm(
            range(batch_start, batch_end),
            desc=f"Batch {batch_num} [{batch_start}-{batch_end}]"
        )

        for idx in pbar:
            sample = dataset[idx]

            image = sample['image']
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert('RGB')
            else:
                image = image.convert('RGB')

            try:
                words, boxes = run_ocr_doctr(image, model)
            except Exception as e:
                print(f"\nOCR failed for sample {idx}: {e}")
                words, boxes = [], []

            processed_sample = {
                'questionId': sample.get('questionId', sample.get('question_id', idx)),
                'question': sample['question'],
                'answers': sample['answers'],
                'image': image,
                'words': words,
                'boxes': boxes,
            }
            batch_samples.append(processed_sample)

            pbar.set_postfix({
                'total': f"{idx+1}/{total}",
                'words': len(words),
            })

        # Save batch
        batch_path = output_path / f'batch_{batch_num:04d}'
        batch_dataset = Dataset.from_list(batch_samples)
        batch_dataset.save_to_disk(str(batch_path))

        checkpoint['processed_batches'].append(str(batch_path))
        checkpoint['last_idx'] = batch_end
        save_checkpoint(checkpoint_path, checkpoint)

        print(f"Saved batch {batch_num} ({len(batch_samples)} samples)")

        del batch_samples
        del batch_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_num += 1

    # Merge batches
    print(f"\nMerging {len(checkpoint['processed_batches'])} batches...")
    all_datasets = []
    for batch_path in sorted(checkpoint['processed_batches']):
        if Path(batch_path).exists():
            all_datasets.append(Dataset.load_from_disk(batch_path))

    if all_datasets:
        final_dataset = concatenate_datasets(all_datasets)
        final_path = output_path / 'final'
        final_dataset.save_to_disk(str(final_path))
        print(f"Saved final dataset ({len(final_dataset)} samples)")

        for batch_path in checkpoint['processed_batches']:
            try:
                import shutil
                shutil.rmtree(batch_path)
            except:
                pass

        checkpoint['complete'] = True
        save_checkpoint(checkpoint_path, checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./data/docvqa_doctr')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'validation'])
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("DocVQA OCR Preprocessing (docTR)")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    # Setup docTR
    model, device = setup_doctr()

    # Load dataset
    print("\nLoading DocVQA dataset...")
    try:
        full_dataset = load_dataset('HuggingFaceM4/DocumentVQA')
        print("Using HuggingFaceM4/DocumentVQA")
    except Exception as e1:
        print(f"HuggingFaceM4 failed: {e1}")
        full_dataset = load_dataset('lmms-lab/DocVQA', 'DocVQA')
        print("Using lmms-lab/DocVQA")

    print(f"Dataset splits: {list(full_dataset.keys())}")

    split_mapping = {
        'train': 'train',
        'validation': 'val' if 'val' in full_dataset else 'validation',
        'test': 'test',
    }

    for split in args.splits:
        source_split = split_mapping.get(split, split)
        if source_split not in full_dataset:
            print(f"Split {source_split} not found, skipping...")
            continue

        dataset = full_dataset[source_split]

        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        process_split(dataset, split, args.output_dir, model, args.batch_size)
        gc.collect()

    print("\n" + "=" * 60)
    print(f"Done! Completed at: {datetime.now()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
