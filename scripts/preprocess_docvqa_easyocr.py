#!/usr/bin/env python3
"""
Pre-process DocVQA dataset with OCR annotations using EasyOCR.

Features:
- Checkpointing: Saves progress every batch, resumes from last checkpoint
- Memory efficient: Processes in batches, clears memory periodically
- GPU accelerated: Uses GPU if available for 10-50x speedup

Usage:
    python scripts/preprocess_docvqa_easyocr.py --output_dir ./data/docvqa
"""

import os
import sys
import gc
import json
import types
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================
# CRITICAL: Patch bidi module BEFORE importing easyocr
# ============================================================
def _patch_bidi():
    """Patch bidi module for English-only OCR on Alliance Canada clusters."""
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
    print("Patched bidi module for English-only OCR")

_patch_bidi()

# Now safe to import
from tqdm import tqdm
import easyocr
from PIL import Image
from datasets import load_dataset, Dataset, concatenate_datasets
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

        # Normalize to 0-1000 range (LayoutLMv3 format)
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


def process_split(dataset, split_name, output_dir, reader, batch_size=500):
    """
    Process a dataset split with checkpointing and memory management.

    Saves batches incrementally to avoid OOM.
    """
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

    # Process in batches
    for batch_start in range(start_idx, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_samples = []

        pbar = tqdm(
            range(batch_start, batch_end),
            desc=f"Batch {batch_num} [{batch_start}-{batch_end}]",
            position=0,
            leave=True
        )

        for idx in pbar:
            sample = dataset[idx]

            # Get image
            image = sample['image']
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert('RGB')
            else:
                image = image.convert('RGB')

            # Run OCR
            try:
                words, boxes = run_ocr(image, reader)
            except Exception as e:
                print(f"\nOCR failed for sample {idx}: {e}")
                words, boxes = [], []

            # Create sample (don't store full image, just serialize)
            processed_sample = {
                'questionId': sample.get('questionId', sample.get('question_id', idx)),
                'question': sample['question'],
                'answers': sample['answers'],
                'image': image,  # Will be serialized by Arrow
                'words': words,
                'boxes': boxes,
            }
            batch_samples.append(processed_sample)

            # Update progress bar
            pbar.set_postfix({
                'total': f"{idx+1}/{total}",
                'mem_gb': f"{get_memory_usage():.1f}"
            })

        # Save batch to disk
        batch_path = output_path / f'batch_{batch_num:04d}'
        batch_dataset = Dataset.from_list(batch_samples)
        batch_dataset.save_to_disk(str(batch_path))

        # Update checkpoint
        checkpoint['processed_batches'].append(str(batch_path))
        checkpoint['last_idx'] = batch_end
        save_checkpoint(checkpoint_path, checkpoint)

        print(f"Saved batch {batch_num} ({len(batch_samples)} samples) to {batch_path}")

        # Clear memory
        del batch_samples
        del batch_dataset
        gc.collect()

        # Force CUDA memory cleanup if using GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        batch_num += 1

    # Merge all batches into final dataset
    print(f"\nMerging {len(checkpoint['processed_batches'])} batches...")
    all_datasets = []
    for batch_path in sorted(checkpoint['processed_batches']):
        if Path(batch_path).exists():
            all_datasets.append(Dataset.load_from_disk(batch_path))

    if all_datasets:
        final_dataset = concatenate_datasets(all_datasets)
        final_path = output_path / 'final'
        final_dataset.save_to_disk(str(final_path))
        print(f"Saved final dataset ({len(final_dataset)} samples) to {final_path}")

        # Cleanup batch files
        for batch_path in checkpoint['processed_batches']:
            try:
                import shutil
                shutil.rmtree(batch_path)
            except:
                pass

        # Update checkpoint to mark complete
        checkpoint['complete'] = True
        save_checkpoint(checkpoint_path, checkpoint)


def get_memory_usage():
    """Get current memory usage in GB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Pre-process DocVQA with EasyOCR")
    parser.add_argument('--output_dir', type=str, default='./data/docvqa',
                        help='Output directory for processed dataset')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'validation'],
                        help='Splits to process')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Samples per batch (for checkpointing)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per split (for testing)')
    args = parser.parse_args()

    print("=" * 60)
    print("DocVQA OCR Preprocessing")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    # Initialize EasyOCR reader
    import torch
    use_gpu = torch.cuda.is_available()
    print(f"\nInitializing EasyOCR reader (GPU: {use_gpu})...")
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    print("EasyOCR reader initialized.")

    # Load DocVQA dataset
    print("\nLoading DocVQA dataset from HuggingFace...")
    try:
        full_dataset = load_dataset('HuggingFaceM4/DocumentVQA')
        print("Using HuggingFaceM4/DocumentVQA")
    except Exception as e1:
        print(f"HuggingFaceM4 failed: {e1}")
        try:
            full_dataset = load_dataset('lmms-lab/DocVQA', 'DocVQA')
            print("Using lmms-lab/DocVQA")
        except Exception as e2:
            print(f"lmms-lab failed: {e2}")
            raise RuntimeError("Could not load DocVQA dataset")

    print(f"Dataset splits: {list(full_dataset.keys())}")

    # Map split names
    split_mapping = {
        'train': 'train',
        'validation': 'val' if 'val' in full_dataset else 'validation',
        'test': 'test',
    }

    # Process each split
    for split in args.splits:
        source_split = split_mapping.get(split, split)
        if source_split not in full_dataset:
            print(f"Split {source_split} not found, skipping...")
            continue

        dataset = full_dataset[source_split]

        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        process_split(dataset, split, args.output_dir, reader, args.batch_size)

        # Force garbage collection between splits
        gc.collect()

    print("\n" + "=" * 60)
    print(f"Done! Completed at: {datetime.now()}")
    print(f"Processed dataset saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
