#!/usr/bin/env python3
"""
Pre-process DocVQA dataset with OCR using streaming mode.
Uses EasyOCR and processes one sample at a time to avoid OOM.
"""

import os
import sys
import gc
import json
import types
import argparse
from pathlib import Path
from datetime import datetime

# Set EasyOCR model directory BEFORE importing easyocr
# This avoids writing to home directory (quota issues on clusters)
EASYOCR_MODEL_DIR = os.environ.get('EASYOCR_MODEL_DIR',
    os.path.join(os.environ.get('SCRATCH', '/tmp'), 'easyocr_models'))
os.makedirs(EASYOCR_MODEL_DIR, exist_ok=True)
os.environ['EASYOCR_MODULE_PATH'] = EASYOCR_MODEL_DIR

# Patch bidi module for compatibility
def _patch_bidi():
    try:
        from bidi import get_display
        return
    except (ImportError, AttributeError):
        pass
    bidi_module = types.ModuleType('bidi')
    def get_display(text, *args, **kwargs):
        return text
    bidi_module.get_display = get_display
    algorithm_module = types.ModuleType('bidi.algorithm')
    algorithm_module.get_display = get_display
    bidi_module.algorithm = algorithm_module
    sys.modules['bidi'] = bidi_module
    sys.modules['bidi.algorithm'] = algorithm_module

_patch_bidi()

from tqdm import tqdm
import easyocr
from PIL import Image
from datasets import load_dataset, Dataset
import numpy as np


def run_ocr(image, reader):
    """Run EasyOCR on an image and return words and boxes."""
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image

    height, width = img_array.shape[:2]
    results = reader.readtext(img_array)

    words = []
    boxes = []

    for (bbox, text, conf) in results:
        if not text.strip():
            continue

        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        box = [
            int(1000 * x1 / width),
            int(1000 * y1 / height),
            int(1000 * x2 / width),
            int(1000 * y2 / height),
        ]
        box = [max(0, min(1000, v)) for v in box]

        words.append(text.strip())
        boxes.append(box)

    return words, boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./data/docvqa')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / '_checkpoint.json'

    # Load checkpoint
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_idx = checkpoint.get('last_idx', 0)
        # Ensure processed_files key exists (for backwards compatibility)
        if 'processed_files' not in checkpoint:
            checkpoint['processed_files'] = []
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        start_idx = args.start_idx
        checkpoint = {'last_idx': start_idx, 'processed_files': []}

    print("=" * 60)
    print("DocVQA OCR Preprocessing (Streaming)")
    print(f"Output: {output_dir}")
    print(f"Starting from: {start_idx}")
    print("=" * 60)

    # Initialize EasyOCR
    import torch
    use_gpu = torch.cuda.is_available()
    print(f"Initializing EasyOCR (GPU: {use_gpu})...")
    print(f"Model storage: {EASYOCR_MODEL_DIR}")
    reader = easyocr.Reader(['en'], gpu=use_gpu, model_storage_directory=EASYOCR_MODEL_DIR)

    # Load dataset in STREAMING mode - doesn't load into memory!
    print("Loading DocVQA dataset (streaming)...")
    ds = load_dataset('HuggingFaceM4/DocumentVQA', split=args.split, streaming=True)

    # Skip to start index
    ds_iter = iter(ds)
    for _ in range(start_idx):
        try:
            next(ds_iter)
        except StopIteration:
            print("Reached end of dataset during skip")
            return

    # Process samples
    batch = []
    idx = start_idx
    max_idx = args.max_samples + start_idx if args.max_samples else float('inf')

    pbar = tqdm(desc="Processing", initial=start_idx)

    for sample in ds_iter:
        if idx >= max_idx:
            break

        try:
            image = sample['image']
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert('RGB')
            else:
                image = image.convert('RGB')

            words, boxes = run_ocr(image, reader)

            processed = {
                'questionId': sample.get('questionId', sample.get('question_id', idx)),
                'question': sample['question'],
                'answers': sample['answers'],
                'words': words,
                'boxes': boxes,
                # Store image bytes instead of PIL for memory efficiency
                'image': image,
            }
            batch.append(processed)

        except Exception as e:
            print(f"\nError at idx {idx}: {e}")
            batch.append({
                'questionId': idx,
                'question': sample.get('question', ''),
                'answers': sample.get('answers', []),
                'words': [],
                'boxes': [],
                'image': None,
            })

        idx += 1
        pbar.update(1)
        pbar.set_postfix({'words': len(words) if 'words' in locals() else 0})

        # Save batch periodically
        if len(batch) >= args.save_every:
            batch_file = output_dir / f'batch_{idx:06d}.arrow'
            batch_ds = Dataset.from_list(batch)
            batch_ds.save_to_disk(str(batch_file))
            print(f"\nSaved batch to {batch_file}")

            checkpoint['last_idx'] = idx
            checkpoint['processed_files'].append(str(batch_file))
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)

            batch = []
            gc.collect()

    # Save remaining
    if batch:
        batch_file = output_dir / f'batch_{idx:06d}.arrow'
        batch_ds = Dataset.from_list(batch)
        batch_ds.save_to_disk(str(batch_file))
        print(f"\nSaved final batch to {batch_file}")

        checkpoint['last_idx'] = idx
        checkpoint['processed_files'].append(str(batch_file))
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)

    pbar.close()
    print(f"\nDone! Processed {idx - start_idx} samples")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
