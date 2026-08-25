#!/usr/bin/env python3
"""Check what fields are available in DocVQA dataset."""

from datasets import load_dataset

print("Loading DocVQA dataset...")
try:
    ds = load_dataset('HuggingFaceM4/DocumentVQA', split='train[:5]')
    print(f"\nHuggingFaceM4/DocumentVQA fields:")
    print(f"  Columns: {ds.column_names}")
    print(f"  Sample keys: {list(ds[0].keys())}")
    for key in ds[0].keys():
        val = ds[0][key]
        if isinstance(val, list) and len(val) > 0:
            print(f"  {key}: list of {len(val)} items, first: {val[0] if len(str(val[0])) < 100 else str(val[0])[:100]}")
        elif isinstance(val, str):
            print(f"  {key}: {val[:100] if len(val) > 100 else val}")
        else:
            print(f"  {key}: {type(val).__name__}")
except Exception as e:
    print(f"HuggingFaceM4 error: {e}")

print("\n" + "="*50 + "\n")

try:
    ds = load_dataset('lmms-lab/DocVQA', 'DocVQA', split='train[:5]')
    print(f"lmms-lab/DocVQA fields:")
    print(f"  Columns: {ds.column_names}")
    print(f"  Sample keys: {list(ds[0].keys())}")
    for key in ds[0].keys():
        val = ds[0][key]
        if isinstance(val, list) and len(val) > 0:
            print(f"  {key}: list of {len(val)} items, first: {val[0] if len(str(val[0])) < 100 else str(val[0])[:100]}")
        elif isinstance(val, str):
            print(f"  {key}: {val[:100] if len(val) > 100 else val}")
        else:
            print(f"  {key}: {type(val).__name__}")
except Exception as e:
    print(f"lmms-lab error: {e}")

print("\n" + "="*50 + "\n")

try:
    ds = load_dataset('nielsr/docvqa_1200_examples_donut', split='train[:5]')
    print(f"nielsr/docvqa_1200_examples_donut fields:")
    print(f"  Columns: {ds.column_names}")
except Exception as e:
    print(f"nielsr error: {e}")
