#!/usr/bin/env python3
"""
Merge DocVQA batch files into a single dataset.
"""

import argparse
from pathlib import Path
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing batch_*.arrow files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for merged dataset')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'merged'

    # Find all batch files
    batch_dirs = sorted(input_dir.glob('batch_*.arrow'))
    print(f"Found {len(batch_dirs)} batch directories")

    if not batch_dirs:
        print("No batch files found!")
        return

    # Load and merge
    datasets = []
    total_samples = 0

    for batch_dir in tqdm(batch_dirs, desc="Loading batches"):
        try:
            ds = Dataset.load_from_disk(str(batch_dir))
            datasets.append(ds)
            total_samples += len(ds)
        except Exception as e:
            print(f"Error loading {batch_dir}: {e}")

    print(f"\nTotal samples: {total_samples}")
    print("Merging datasets...")

    merged = concatenate_datasets(datasets)
    print(f"Merged dataset size: {len(merged)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_dir}...")
    merged.save_to_disk(str(output_dir))

    print(f"\nDone! Merged dataset saved to: {output_dir}")
    print(f"Columns: {merged.column_names}")


if __name__ == '__main__':
    main()
