#!/usr/bin/env python3
"""
Pre-process DocVQA dataset with OCR annotations.

This script downloads the DocVQA dataset, runs OCR on all images,
and saves the results with OCR annotations included.

Usage:
    python scripts/preprocess_docvqa_ocr.py --output_dir ./data/docvqa_ocr
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pytesseract
from PIL import Image
from datasets import load_dataset, Dataset, DatasetDict

def run_ocr(image):
    """Run pytesseract OCR on an image and return words and boxes."""
    # Get OCR data with bounding boxes
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    boxes = []

    width, height = image.size

    for i, word in enumerate(ocr_data['text']):
        # Skip empty words
        if not word.strip():
            continue

        # Get bounding box
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]

        # Normalize to 0-1000 range (LayoutLMv3 format)
        box = [
            int(1000 * x / width),
            int(1000 * y / height),
            int(1000 * (x + w) / width),
            int(1000 * (y + h) / height),
        ]

        # Clamp values
        box = [max(0, min(1000, v)) for v in box]

        words.append(word.strip())
        boxes.append(box)

    return words, boxes


def process_split(dataset, split_name, output_dir):
    """Process a dataset split and add OCR annotations."""
    print(f"\nProcessing {split_name} split ({len(dataset)} samples)...")

    processed_samples = []

    for idx, sample in enumerate(tqdm(dataset, desc=f"OCR {split_name}")):
        # Get image
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')

        # Run OCR
        try:
            words, boxes = run_ocr(image)
        except Exception as e:
            print(f"OCR failed for sample {idx}: {e}")
            words, boxes = [], []

        # Create new sample with OCR data
        processed_sample = {
            'questionId': sample.get('questionId', sample.get('question_id', idx)),
            'question': sample['question'],
            'answers': sample['answers'],
            'image': image,
            'words': words,
            'boxes': boxes,
        }

        processed_samples.append(processed_sample)

    # Save as Arrow dataset
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to Dataset and save
    processed_dataset = Dataset.from_list(processed_samples)
    processed_dataset.save_to_disk(str(output_path))

    print(f"Saved {len(processed_samples)} samples to {output_path}")

    return processed_dataset


def main():
    parser = argparse.ArgumentParser(description="Pre-process DocVQA with OCR")
    parser.add_argument('--output_dir', type=str, default='./data/docvqa',
                        help='Output directory for processed dataset')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'validation'],
                        help='Splits to process')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per split (for testing)')
    args = parser.parse_args()

    # Check if pytesseract is available
    try:
        pytesseract.get_tesseract_version()
        print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    except Exception as e:
        print(f"Error: Tesseract not available: {e}")
        print("Please install tesseract: sudo apt install tesseract-ocr")
        return

    # Load DocVQA dataset - HuggingFaceM4 has train/validation/test splits
    print("Loading DocVQA dataset from HuggingFace...")
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
            full_dataset = load_dataset('eliolio/docvqa')
            print("Using eliolio/docvqa")

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

        process_split(dataset, split, args.output_dir)

    print("\nDone! Upload the processed data to the cluster:")
    print(f"  rsync -avz {args.output_dir}/ cluster:/scratch/user/calidoc/data/docvqa/")


if __name__ == '__main__':
    main()
