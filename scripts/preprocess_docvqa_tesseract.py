#!/usr/bin/env python3
"""
Pre-process DocVQA dataset with OCR annotations using Tesseract.

Alternative to EasyOCR for Alliance Canada where EasyOCR can't be installed.
Tesseract is available via module: `module load tesseract`

Features:
- Checkpointing: Saves progress every batch, resumes from last checkpoint
- Memory efficient: Processes in batches, clears memory periodically
- Uses Tesseract OCR (available on Alliance Canada via module)

Usage:
    python scripts/preprocess_docvqa_tesseract.py --output_dir ./data/docvqa
"""

import os
import sys
import gc
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from io import BytesIO
import tempfile

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np


def check_tesseract():
    """Check if Tesseract is available."""
    try:
        result = subprocess.run(
            ['tesseract', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"Tesseract available: {version}")
            return True
    except FileNotFoundError:
        pass

    print("ERROR: Tesseract not found. On Alliance Canada, run: module load tesseract")
    return False


def run_ocr_tesseract(image):
    """
    Run Tesseract OCR on an image and return words and boxes.

    Uses `tesseract ... tsv` output format for word-level bounding boxes.
    """
    if isinstance(image, Image.Image):
        img = image.convert('RGB')
    else:
        img = Image.open(image).convert('RGB')

    width, height = img.size

    # Save image to temp file (Tesseract needs a file)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Run tesseract with TSV output for bounding boxes
        result = subprocess.run(
            ['tesseract', tmp_path, 'stdout', '-l', 'eng', 'tsv'],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"Tesseract error: {result.stderr}")
            return [], []

        # Parse TSV output
        words = []
        boxes = []

        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # Only header or empty
            return [], []

        # TSV columns: level, page_num, block_num, par_num, line_num, word_num,
        #              left, top, width, height, conf, text
        for line in lines[1:]:  # Skip header
            parts = line.split('\t')
            if len(parts) < 12:
                continue

            try:
                conf = float(parts[10])
                text = parts[11].strip()

                # Skip low confidence or empty text
                if conf < 0 or not text:
                    continue

                left = int(parts[6])
                top = int(parts[7])
                w = int(parts[8])
                h = int(parts[9])

                # Convert to x1, y1, x2, y2
                x1, y1 = left, top
                x2, y2 = left + w, top + h

                # Normalize to 0-1000 range (LayoutLMv3 format)
                box = [
                    int(1000 * x1 / width),
                    int(1000 * y1 / height),
                    int(1000 * x2 / width),
                    int(1000 * y2 / height),
                ]
                box = [max(0, min(1000, v)) for v in box]

                words.append(text)
                boxes.append(box)

            except (ValueError, IndexError) as e:
                continue

        return words, boxes

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


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


def process_split(dataset, split_name, output_dir, batch_size=500, save_every=500):
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
                words, boxes = run_ocr_tesseract(image)
            except Exception as e:
                print(f"\nOCR failed for sample {idx}: {e}")
                words, boxes = [], []

            # Create sample
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
                'words': len(words),
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
    parser = argparse.ArgumentParser(description="Pre-process DocVQA with Tesseract OCR")
    parser.add_argument('--output_dir', type=str, default='./data/docvqa',
                        help='Output directory for processed dataset')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'validation'],
                        help='Splits to process')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Samples per batch (for checkpointing)')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N samples')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per split (for testing)')
    args = parser.parse_args()

    print("=" * 60)
    print("DocVQA OCR Preprocessing (Tesseract)")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    # Check Tesseract availability
    if not check_tesseract():
        sys.exit(1)

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

        process_split(dataset, split, args.output_dir, args.batch_size, args.save_every)

        # Force garbage collection between splits
        gc.collect()

    print("\n" + "=" * 60)
    print(f"Done! Completed at: {datetime.now()}")
    print(f"Processed dataset saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
