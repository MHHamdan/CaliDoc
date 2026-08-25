#!/usr/bin/env python3
"""
Entity-Level Conformal Prediction Evaluation

Addresses:
  - Reviewer weakness #2: no entity-level CP analysis
  - Reviewer Q4: BIO consistency in prediction sets

Computes:
  1. Groups consecutive BIO tokens into entities (B-X followed by I-X of same type)
  2. Entity covered = ALL tokens in entity have true label covered
  3. BIO consistency: counts prediction sets with conflicting BIO labels
  4. Reconstructs LAC coverage from confidence + threshold

Inputs: per-token prediction JSONs from results/conformal_v2_multiseed/
Outputs: entity-level vs token-level coverage table
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from calidoc.data.funsd_dataset import FUNSD_LABELS
from calidoc.data.cord_dataset import CORD_LABELS

DATASET_LABELS = {
    'funsd': FUNSD_LABELS,
    'cord': CORD_LABELS,
}


def load_per_token_predictions(json_path: str):
    """Load per-token prediction JSON."""
    with open(json_path) as f:
        documents = json.load(f)
    return documents


def get_entity_type(label_str: str):
    """Extract entity type from BIO label (e.g., 'B-HEADER' -> 'HEADER')."""
    if label_str == 'O':
        return None
    if label_str.startswith('B-') or label_str.startswith('I-'):
        return label_str[2:]
    return None


def get_bio_prefix(label_str: str):
    """Extract BIO prefix from label."""
    if label_str == 'O':
        return 'O'
    if label_str.startswith('B-'):
        return 'B'
    if label_str.startswith('I-'):
        return 'I'
    return 'O'


def extract_entities(tokens):
    """Extract entities from a sequence of tokens using BIO tags.

    An entity is a B-X tag followed by zero or more I-X tags of the same type.
    O tokens are skipped (not entities).

    Returns list of entities, each is a dict with:
      - entity_type: str (e.g., 'HEADER', 'MENU.NM')
      - tokens: list of token dicts
      - token_indices: list of indices into the original token list
    """
    entities = []
    current_entity = None

    for i, tok in enumerate(tokens):
        label = tok['true_label']
        prefix = get_bio_prefix(label)
        etype = get_entity_type(label)

        if prefix == 'B' and etype is not None:
            # Start new entity
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {
                'entity_type': etype,
                'tokens': [tok],
                'token_indices': [i],
            }
        elif prefix == 'I' and etype is not None:
            # Continue entity if same type
            if current_entity is not None and current_entity['entity_type'] == etype:
                current_entity['tokens'].append(tok)
                current_entity['token_indices'].append(i)
            else:
                # I-tag without matching B-tag; treat as new entity
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {
                    'entity_type': etype,
                    'tokens': [tok],
                    'token_indices': [i],
                }
        else:
            # O token: close any open entity
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    return entities


def compute_lac_threshold_from_results(results_json_path: str, coverage_level: str = 'coverage_90'):
    """Extract LAC threshold from results JSON."""
    with open(results_json_path) as f:
        results = json.load(f)

    # Try softmax LAC first
    threshold = None
    if 'conformal_softmax' in results:
        lac = results['conformal_softmax'].get('lac', {})
        if coverage_level in lac:
            threshold = lac[coverage_level].get('threshold')

    return threshold


def reconstruct_lac_coverage(token, threshold):
    """Reconstruct whether a token is covered under LAC.

    LAC: token is covered if nonconformity score <= threshold.
    Nonconformity score = 1 - softmax(true_class).

    If pred == true, score = 1 - confidence.
    If pred != true, score >= 1 - confidence (conservative: score = 1.0).
    """
    if token['pred_label'] == token['true_label']:
        score = 1.0 - token['confidence']
    else:
        score = 1.0
    return score <= threshold


def check_bio_consistency(prediction_set, label_list):
    """Check if a prediction set contains BIO-inconsistent labels.

    Inconsistencies:
    1. Contains both B-X and B-Y for different entity types X, Y
       (ambiguous entity start)
    2. Contains B-X and I-Y where X != Y (conflicting continuation)
    3. Contains I-X without any B-X (orphan continuation)

    Returns dict with consistency flags.
    """
    if not prediction_set:
        return {'consistent': True, 'issues': []}

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    set_labels = []
    for lid in prediction_set:
        if lid in id2label:
            set_labels.append(id2label[lid])

    b_types = set()
    i_types = set()
    has_o = False

    for label in set_labels:
        if label == 'O':
            has_o = True
        elif label.startswith('B-'):
            b_types.add(label[2:])
        elif label.startswith('I-'):
            i_types.add(label[2:])

    issues = []

    # Multiple B-tags of different types
    if len(b_types) > 1:
        issues.append(f"multiple_b_types:{','.join(sorted(b_types))}")

    # I-tag without matching B-tag
    orphan_i = i_types - b_types
    if orphan_i:
        issues.append(f"orphan_i_types:{','.join(sorted(orphan_i))}")

    # B-X and I-Y where X != Y
    if b_types and i_types:
        conflicting = i_types - b_types
        if conflicting:
            issues.append(f"conflicting_bi:{','.join(sorted(conflicting))}")

    return {
        'consistent': len(issues) == 0,
        'issues': issues,
        'n_b_types': len(b_types),
        'n_i_types': len(i_types),
        'has_o': has_o,
    }


def run_analysis(dataset_name, base_dir, seeds, alpha=0.10):
    """Run entity-level CP analysis."""
    label_list = DATASET_LABELS[dataset_name]
    all_seed_results = {}

    for seed in seeds:
        json_path = os.path.join(
            base_dir, dataset_name, f'seed_{seed}',
            f'per_token_predictions_{dataset_name}.json')

        # Find results JSON for threshold
        seed_dir = os.path.join(base_dir, dataset_name, f'seed_{seed}')
        results_files = [f for f in os.listdir(seed_dir) if f.startswith('results_v2')]
        if not results_files:
            print(f"  [SKIP] No results_v2 file for seed {seed}")
            continue
        results_path = os.path.join(seed_dir, results_files[0])

        if not os.path.exists(json_path):
            print(f"  [SKIP] {json_path} not found")
            continue

        documents = load_per_token_predictions(json_path)
        threshold = compute_lac_threshold_from_results(results_path, 'coverage_90')

        if threshold is None:
            print(f"  [SKIP] No LAC threshold found for seed {seed}")
            continue

        print(f"\n  Seed {seed}: {len(documents)} documents, LAC threshold={threshold:.8f}")

        # Process each document
        total_tokens = 0
        total_token_covered = 0
        total_entities = 0
        total_entity_covered = 0
        entity_type_stats = defaultdict(lambda: {
            'total': 0, 'covered': 0, 'n_tokens': 0})
        entity_length_stats = defaultdict(lambda: {'total': 0, 'covered': 0})

        # BIO consistency stats
        total_bio_checks = 0
        total_bio_inconsistent = 0
        bio_issue_counts = defaultdict(int)

        # Use APS prediction sets for BIO consistency (since LAC sets are often empty)
        aps_bio_checks = 0
        aps_bio_inconsistent = 0

        for doc in documents:
            tokens = doc['tokens']

            # Token-level coverage (reconstructed LAC)
            for tok in tokens:
                total_tokens += 1
                if reconstruct_lac_coverage(tok, threshold):
                    total_token_covered += 1

            # Entity-level coverage
            entities = extract_entities(tokens)
            for entity in entities:
                total_entities += 1
                entity_type_stats[entity['entity_type']]['total'] += 1
                entity_type_stats[entity['entity_type']]['n_tokens'] += len(entity['tokens'])

                # Entity covered = all tokens covered
                all_covered = all(
                    reconstruct_lac_coverage(t, threshold)
                    for t in entity['tokens'])
                if all_covered:
                    total_entity_covered += 1
                    entity_type_stats[entity['entity_type']]['covered'] += 1

                # Entity length stats
                elen = len(entity['tokens'])
                entity_length_stats[elen]['total'] += 1
                if all_covered:
                    entity_length_stats[elen]['covered'] += 1

            # BIO consistency on APS prediction sets
            for tok in tokens:
                aps_set = tok.get('prediction_set_aps', [])
                if aps_set:
                    aps_bio_checks += 1
                    consistency = check_bio_consistency(aps_set, label_list)
                    if not consistency['consistent']:
                        aps_bio_inconsistent += 1
                        for issue in consistency['issues']:
                            bio_issue_counts[issue] += 1

        # Compile results
        token_coverage = total_token_covered / total_tokens if total_tokens > 0 else 0.0
        entity_coverage = total_entity_covered / total_entities if total_entities > 0 else 0.0
        aps_inconsistency_rate = (aps_bio_inconsistent / aps_bio_checks
                                  if aps_bio_checks > 0 else 0.0)

        per_type_results = {}
        for etype in sorted(entity_type_stats.keys()):
            s = entity_type_stats[etype]
            per_type_results[etype] = {
                'total': s['total'],
                'covered': s['covered'],
                'coverage': s['covered'] / s['total'] if s['total'] > 0 else 0.0,
                'avg_tokens': s['n_tokens'] / s['total'] if s['total'] > 0 else 0.0,
            }

        per_length_results = {}
        for elen in sorted(entity_length_stats.keys()):
            s = entity_length_stats[elen]
            per_length_results[elen] = {
                'total': s['total'],
                'covered': s['covered'],
                'coverage': s['covered'] / s['total'] if s['total'] > 0 else 0.0,
            }

        seed_result = {
            'seed': seed,
            'lac_threshold': threshold,
            'n_tokens': total_tokens,
            'n_entities': total_entities,
            'token_coverage': token_coverage,
            'entity_coverage': entity_coverage,
            'coverage_gap': token_coverage - entity_coverage,
            'aps_bio_inconsistency_rate': aps_inconsistency_rate,
            'aps_bio_checks': aps_bio_checks,
            'aps_bio_inconsistent': aps_bio_inconsistent,
            'bio_issue_counts': dict(bio_issue_counts),
            'per_entity_type': per_type_results,
            'per_entity_length': per_length_results,
        }

        all_seed_results[seed] = seed_result

    return all_seed_results


def print_table(results, dataset_name):
    """Print formatted results table."""
    seeds = sorted(results.keys())

    print(f"\n{'='*80}")
    print(f"ENTITY-LEVEL CP RESULTS: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Overview table
    print(f"\n{'Metric':<35} ", end='')
    for seed in seeds:
        print(f"Seed {seed:>5}  ", end='')
    if len(seeds) > 1:
        print(f"  Mean±Std")
    else:
        print()

    print("-" * (35 + 12 * len(seeds) + (15 if len(seeds) > 1 else 0)))

    metrics = [
        ('Token-level coverage', 'token_coverage'),
        ('Entity-level coverage', 'entity_coverage'),
        ('Coverage gap (tok - ent)', 'coverage_gap'),
        ('APS BIO inconsistency rate', 'aps_bio_inconsistency_rate'),
    ]

    for name, key in metrics:
        print(f"{name:<35} ", end='')
        vals = []
        for seed in seeds:
            v = results[seed][key]
            vals.append(v)
            print(f"{v*100:>6.1f}%  ", end='')
        if len(seeds) > 1:
            print(f"  {np.mean(vals)*100:.1f}%±{np.std(vals)*100:.1f}%")
        else:
            print()

    print(f"{'# Entities':<35} ", end='')
    for seed in seeds:
        print(f"{results[seed]['n_entities']:>7}  ", end='')
    print()

    print(f"{'# Tokens':<35} ", end='')
    for seed in seeds:
        print(f"{results[seed]['n_tokens']:>7}  ", end='')
    print()

    # Per entity-type table (averaged across seeds)
    print(f"\n{'='*80}")
    print(f"PER-ENTITY-TYPE COVERAGE: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Collect all entity types
    all_types = set()
    for seed in seeds:
        all_types.update(results[seed]['per_entity_type'].keys())

    print(f"{'Entity Type':<30} {'n_ent':>6} {'Avg Tok':>7} {'Coverage':>8}")
    print("-" * 55)

    for etype in sorted(all_types):
        n_total = int(np.mean([
            results[s]['per_entity_type'].get(etype, {}).get('total', 0)
            for s in seeds]))
        avg_tok = np.mean([
            results[s]['per_entity_type'].get(etype, {}).get('avg_tokens', 0)
            for s in seeds])
        cov = np.mean([
            results[s]['per_entity_type'].get(etype, {}).get('coverage', 0.0)
            for s in seeds])
        print(f"{etype:<30} {n_total:>6} {avg_tok:>7.1f} {cov*100:>7.1f}%")

    # Coverage by entity length
    print(f"\n{'='*80}")
    print(f"COVERAGE BY ENTITY LENGTH: {dataset_name.upper()}")
    print(f"{'='*80}")

    all_lengths = set()
    for seed in seeds:
        all_lengths.update(results[seed]['per_entity_length'].keys())

    print(f"{'Length':>6} {'n_ent':>6} {'Coverage':>8}")
    print("-" * 24)

    for elen in sorted(all_lengths):
        n_total = int(np.mean([
            results[s]['per_entity_length'].get(elen, {}).get('total', 0)
            for s in seeds]))
        cov = np.mean([
            results[s]['per_entity_length'].get(elen, {}).get('coverage', 0.0)
            for s in seeds])
        if n_total > 0:
            print(f"{elen:>6} {n_total:>6} {cov*100:>7.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Entity-Level Conformal Prediction Evaluation")
    parser.add_argument('--base_dir', type=str,
                        default='results/conformal_v2_multiseed',
                        help='Base directory with per-token prediction JSONs')
    parser.add_argument('--datasets', nargs='+', default=['funsd', 'cord'],
                        choices=['funsd', 'cord'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--output_dir', type=str,
                        default='results/entity_level_cp',
                        help='Output directory for JSON results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")

        results = run_analysis(dataset_name, args.base_dir, args.seeds, args.alpha)
        if results:
            print_table(results, dataset_name)
            all_results[dataset_name] = {
                str(k): v for k, v in results.items()}

    # Save results
    out_path = os.path.join(args.output_dir, 'entity_level_cp_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
