#!/bin/bash
# Download all datasets for CaliDoc
# Run this locally before transferring to cluster

set -e

DATA_DIR="${1:-./data}"

echo "Downloading datasets to: $DATA_DIR"
echo ""

python3 << 'EOF'
import os
import sys
from pathlib import Path
from datasets import load_dataset

data_dir = os.environ.get("DATA_DIR", "./data")
data_dir = Path(data_dir)

datasets_to_download = [
    ("nielsr/docvqa", "docvqa", ["train", "validation", "test"]),
    ("nielsr/funsd", "funsd", ["train", "test"]),
    ("naver-clova-ix/cord-v2", "cord", ["train", "validation", "test"]),
]

for hf_name, local_name, splits in datasets_to_download:
    print(f"\n{'='*50}")
    print(f"Downloading {hf_name}")
    print(f"{'='*50}")

    local_dir = data_dir / local_name
    local_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_dir = local_dir / split
        if split_dir.exists() and (split_dir / "dataset_info.json").exists():
            print(f"  {split}: already exists, skipping")
            continue

        print(f"  {split}: downloading...")
        try:
            ds = load_dataset(hf_name, split=split)
            ds.save_to_disk(str(split_dir))
            print(f"  {split}: saved to {split_dir}")
        except Exception as e:
            print(f"  {split}: ERROR - {e}")

print("\nDownload complete!")
EOF

echo ""
echo "Done! Data saved to: $DATA_DIR"
