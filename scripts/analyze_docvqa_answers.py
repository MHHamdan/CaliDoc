#!/usr/bin/env python3
"""
Analyze DocVQA to check what percentage of answers are extractable spans.

This helps diagnose why extractive QA approach fails on DocVQA.
"""

import argparse
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm


def check_answer_in_text(answer: str, words: list) -> bool:
    """Check if answer appears as substring in OCR text."""
    if not words or not answer:
        return False
    text = ' '.join(words).lower()
    return answer.lower().strip() in text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/scratch/$USER/calidoc/data/docvqa/train_merged',
                        help='Path to preprocessed DocVQA data')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Max samples to analyze')
    args = parser.parse_args()

    # Expand $USER
    import os
    data_dir = args.data_dir.replace('$USER', os.environ.get('USER', ''))

    print(f"Loading dataset from: {data_dir}")
    ds = load_from_disk(data_dir)
    print(f"Total samples: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    # Analyze answer extractability
    found_count = 0
    not_found_count = 0
    no_ocr_count = 0

    n_samples = min(args.max_samples, len(ds))

    print(f"\nAnalyzing {n_samples} samples...")
    examples_not_found = []

    for i in tqdm(range(n_samples)):
        sample = ds[i]
        words = sample.get('words', sample.get('ocr_tokens', []))
        answers = sample.get('answers', [])
        answer = answers[0] if answers else ''

        if not words:
            no_ocr_count += 1
            continue

        if check_answer_in_text(answer, words):
            found_count += 1
        else:
            not_found_count += 1
            if len(examples_not_found) < 5:
                examples_not_found.append({
                    'question': sample.get('question', ''),
                    'answer': answer,
                    'ocr_sample': ' '.join(words[:50]) if words else 'N/A'
                })

    # Report results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Samples analyzed: {n_samples}")
    print(f"Answers found in OCR text: {found_count} ({100*found_count/n_samples:.1f}%)")
    print(f"Answers NOT found: {not_found_count} ({100*not_found_count/n_samples:.1f}%)")
    print(f"No OCR data: {no_ocr_count}")

    print(f"\n{'='*60}")
    print("EXAMPLES WHERE ANSWER NOT FOUND IN OCR TEXT:")
    print(f"{'='*60}")
    for i, ex in enumerate(examples_not_found):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {ex['question'][:100]}...")
        print(f"A: {ex['answer']}")
        print(f"OCR: {ex['ocr_sample']}...")

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    if not_found_count / n_samples > 0.3:
        print("❌ Extractive QA is NOT suitable for DocVQA!")
        print("   Most answers are not verbatim spans in the document.")
        print("   Recommendations:")
        print("   1. Use generative approach (T5, LLM)")
        print("   2. Use answer classification over candidates")
        print("   3. Focus on NER tasks for conformal prediction paper")
    else:
        print("✓ Extractive QA may work for DocVQA")


if __name__ == '__main__':
    main()
