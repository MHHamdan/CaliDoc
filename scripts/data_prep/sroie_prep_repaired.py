#!/usr/bin/env python3
"""
E7 step 2 -- build a REPAIRED SROIE bundle from the official Task-3 sources.

Motivation (see scripts/data_prep/sroie_alignment_diagnosis.py): the
pre-registered strict gate reaches only 94.45% because Task-3 field strings and
the OCR transcription disagree on PUNCTUATION -- overwhelmingly in multi-line
ADDRESS fields -- and on currency/decimal formatting in TOTAL. Neither is OCR
error; both are formatting conventions. Relaxing exactly those two things
lifts alignment to 97.38%, clearing the 95% gate.

This is a NEW, SEPARATELY REPORTED lineage. It does not retroactively edit the
pre-registered exclusion decision (results/sroie_gate_report_20260705.md stays
as recorded), and it is not the upstream pre-tagged bundle either.

Matcher escalation, applied per field in this order, with the winning matcher
recorded per instance so the paper can state exactly how many labels came from
relaxed matching:

  strict       whitespace-strip + casefold, contiguous token span (the gate)
  punctuation  additionally ignore all non-alphanumeric characters
  numeric      TOTAL only: match the token whose parsed number equals the
               field value, tolerating currency symbols and 9 / 9.0 / 9.00

Deliberately NOT used: non-contiguous multi-line matching. It recovers only
0.10% of instances and can stitch together unrelated tokens, which would inject
exactly the kind of silent noise this whole paper is about.

Output goes to a SEPARATE bundle root so that dataset_name stays 'sroie' and no
dataset/loader code has to change:

    <out_root>/sroie/processed/{train,test}.parquet, label_list.json
    <out_root>/sroie/images/{train,test}/...
    <out_root>/splits/sroie_val_es.json, sroie_calib_seed{42,123,456}.json
    <out_root>/PROVENANCE_sroie_repaired_<date>.md

Usage:
  python scripts/data_prep/sroie_prep_repaired.py \
      --root data/SROIE_official/SROIE2019 \
      --out_root data_bundle_sroie_repaired
"""

import argparse
import datetime
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from sroie_prep_gate import (  # noqa: E402
    FIELDS, GATE_THRESHOLD, SROIE_LABELS, VAL_ES_SEED, VAL_ES_SIZE,
    find_span, norm_chars, normalize_bbox, parse_ocr_file)

CALIB_SEEDS = (42, 123, 456)
CALIB_FRACTION = 0.70


def norm_alnum(s):
    return re.sub(r'[^0-9a-z]', '', str(s).casefold())


def as_number(s):
    m = re.findall(r'\d+(?:[.,]\d+)?', str(s).replace(' ', ''))
    if not m:
        return None
    try:
        return round(float(m[-1].replace(',', '.')), 2)
    except ValueError:
        return None


def find_span_alnum(tokens_alnum, entity):
    target = norm_alnum(entity)
    if not target:
        return None
    n, tlen = len(tokens_alnum), len(target)
    for start in range(n):
        if not tokens_alnum[start]:
            continue
        acc = 0
        for end in range(start, n):
            tok = tokens_alnum[end]
            if tok != target[acc:acc + len(tok)]:
                break
            acc += len(tok)
            if acc == tlen:
                return (start, end)
            if acc > tlen:
                break
    return None


def find_numeric(tokens, entity):
    want = as_number(entity)
    if want is None:
        return None
    for i, tok in enumerate(tokens):
        if as_number(tok) == want:
            return (i, i)
    return None


def align_field(field, value, tokens, tokens_norm, tokens_alnum):
    """Returns (span, matcher) or (None, None)."""
    span = find_span(tokens_norm, value)
    if span is not None:
        return span, 'strict'
    span = find_span_alnum(tokens_alnum, value)
    if span is not None:
        return span, 'punctuation'
    if field == 'total':
        span = find_numeric(tokens, value)
        if span is not None:
            return span, 'numeric'
    return None, None


def process_split(split, ocr_dir, ent_dir, img_dir):
    rows = []
    stats = {'receipts': 0, 'entities_total': 0, 'entities_aligned': 0,
             'by_matcher': Counter(), 'failures_by_field': Counter(),
             'failure_examples': defaultdict(list),
             'overlaps_skipped': 0}

    for ocr_path in sorted(ocr_dir.glob('*.txt')):
        stem = ocr_path.stem
        ent_path = ent_dir / f'{stem}.txt'
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            cand = img_dir / f'{stem}{ext}'
            if cand.exists():
                img_path = cand
                break
        if not ent_path.exists() or img_path is None:
            continue

        stats['receipts'] += 1
        tokens, boxes = parse_ocr_file(ocr_path)
        tokens_norm = [norm_chars(t) for t in tokens]
        tokens_alnum = [norm_alnum(t) for t in tokens]
        entities = json.loads(ent_path.read_text(encoding='utf-8', errors='replace'))

        tags = ['O'] * len(tokens)
        claimed = [False] * len(tokens)
        for f in FIELDS:
            value = entities.get(f, '')
            if not str(value).strip():
                continue
            stats['entities_total'] += 1
            span, matcher = align_field(f, str(value), tokens, tokens_norm, tokens_alnum)
            if span is None:
                stats['failures_by_field'][f] += 1
                if len(stats['failure_examples'][f]) < 5:
                    stats['failure_examples'][f].append(
                        {'doc': stem, 'value': str(value)[:80]})
                continue
            s, e = span
            if any(claimed[s:e + 1]):
                # Two fields resolving to overlapping tokens would silently
                # corrupt one of them -- exactly the failure mode under study.
                stats['overlaps_skipped'] += 1
                stats['failures_by_field'][f] += 1
                continue
            stats['entities_aligned'] += 1
            stats['by_matcher'][matcher] += 1
            tags[s] = f'B-{f.upper()}'
            for i in range(s + 1, e + 1):
                tags[i] = f'I-{f.upper()}'
            for i in range(s, e + 1):
                claimed[i] = True

        rows.append({'stem': stem, 'img_src': img_path,
                     'tokens': tokens, 'boxes': boxes, 'tags': tags})

    stats['alignment_rate'] = (stats['entities_aligned']
                               / max(1, stats['entities_total']))
    return rows, stats


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/SROIE_official/SROIE2019'))
    ap.add_argument('--out_root', type=Path, default=Path('data_bundle_sroie_repaired'))
    ap.add_argument('--allow_gate_failure', action='store_true')
    args = ap.parse_args()

    aligned, all_stats = {}, {}
    for split in ('train', 'test'):
        rows, stats = process_split(split, args.root / split / 'box',
                                    args.root / split / 'entities',
                                    args.root / split / 'img')
        aligned[split], all_stats[split] = rows, stats
        print(f'{split}: {stats["receipts"]} receipts, '
              f'alignment {stats["alignment_rate"]:.4%} '
              f'({stats["entities_aligned"]}/{stats["entities_total"]}) '
              f'matchers={dict(stats["by_matcher"])} '
              f'overlaps_skipped={stats["overlaps_skipped"]}')

    failed = [s for s in all_stats if all_stats[s]['alignment_rate'] < GATE_THRESHOLD]
    if failed and not args.allow_gate_failure:
        print(f'\nGATE FAILED on {failed} (< {GATE_THRESHOLD:.0%}); writing nothing.')
        for s in failed:
            for f, c in all_stats[s]['failures_by_field'].most_common():
                print(f'  {s}/{f}: {c} failures')
        sys.exit(1)
    print(f'\nREPAIRED GATE PASSED (>= {GATE_THRESHOLD:.0%} in every split).')

    out_root = args.out_root
    out_dir = out_root / 'sroie' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ('doc_id', pa.string()),
        ('tokens', pa.list_(pa.string())),
        ('bboxes', pa.list_(pa.list_(pa.int64(), 4))),
        ('ner_tags', pa.list_(pa.string())),
        ('image_path', pa.string()),
        ('split', pa.string()),
        ('ocr_confidence', pa.list_(pa.float64())),
    ])

    split_stats = {}
    for split, rows in aligned.items():
        (out_root / 'sroie' / 'images' / split).mkdir(parents=True, exist_ok=True)
        out_rows = []
        for r in rows:
            width, height = Image.open(r['img_src']).size
            ext = r['img_src'].suffix.lower()
            doc_id = f'sroie_{split}_{r["stem"]}'
            rel_img = f'sroie/images/{split}/{doc_id}{ext}'
            shutil.copy2(r['img_src'], out_root / rel_img)
            out_rows.append({
                'doc_id': doc_id,
                'tokens': r['tokens'],
                'bboxes': [normalize_bbox(b, width, height) for b in r['boxes']],
                'ner_tags': r['tags'],
                'image_path': rel_img,
                'split': split,
                'ocr_confidence': None,
            })
        pq.write_table(pa.Table.from_pylist(out_rows, schema=schema),
                       out_dir / f'{split}.parquet')
        n_tok = sum(len(r['tokens']) for r in rows)
        n_o = sum(t == 'O' for r in rows for t in r['tags'])
        split_stats[split] = {'docs': len(rows), 'tokens': n_tok,
                              'o_fraction': n_o / max(1, n_tok),
                              'alignment_rate': all_stats[split]['alignment_rate']}
        print(f'  wrote {split}.parquet: {len(rows)} docs, {n_tok} tokens, '
              f'O={split_stats[split]["o_fraction"]:.4f}')

    matcher_totals = Counter()
    for s in all_stats.values():
        matcher_totals.update(s['by_matcher'])
    (out_dir / 'label_list.json').write_text(json.dumps({
        'labels': SROIE_LABELS, 'num_bio_labels': 9, 'num_fields': 4,
        'expected_o_fraction': {k: v['o_fraction'] for k, v in split_stats.items()},
        'label_provenance': (
            'REPAIRED ALIGNMENT from the official ICDAR-2019 Task-3 field '
            'strings. Matcher escalation strict -> punctuation-insensitive -> '
            'numeric (TOTAL only); non-contiguous matching deliberately NOT '
            'used. Counts by matcher: '
            + json.dumps(dict(matcher_totals))
            + '. Alignment rate: '
            + json.dumps({k: round(v['alignment_rate'], 6)
                          for k, v in split_stats.items()})
            + '. This is a SEPARATE lineage from both the upstream pre-tagged '
              'bundle and the strict-gate exclusion decision, which stands as '
              'recorded. PENDING PI SIGN-OFF.'),
    }, indent=2))

    (out_root / 'splits').mkdir(parents=True, exist_ok=True)
    train_ids = sorted(f'sroie_train_{r["stem"]}' for r in aligned['train'])
    val_es = sorted(random.Random(VAL_ES_SEED).sample(train_ids, VAL_ES_SIZE))
    (out_root / 'splits' / 'sroie_val_es.json').write_text(json.dumps({
        'dataset': 'sroie', 'source_split': 'train', 'seed': VAL_ES_SEED,
        'purpose': 'early stopping ONLY; these docs must be REMOVED from training',
        'effective_train_size': len(train_ids) - VAL_ES_SIZE,
        'doc_ids': val_es,
    }, indent=2))

    test_ids = sorted(f'sroie_test_{r["stem"]}' for r in aligned['test'])
    for seed in CALIB_SEEDS:
        ids = list(test_ids)
        random.Random(seed).shuffle(ids)
        k = int(round(CALIB_FRACTION * len(ids)))
        (out_root / 'splits' / f'sroie_calib_seed{seed}.json').write_text(json.dumps({
            'dataset': 'sroie', 'source_split': 'test', 'seed': seed,
            'purpose': 'document-level 70/30 conformal calibration/evaluation split',
            'calibration': sorted(ids[:k]),
            'evaluation': sorted(ids[k:]),
        }, indent=2))

    today = datetime.date.today().strftime('%Y%m%d')
    checks = {}
    for p in sorted(out_dir.glob('*.parquet')) + [out_dir / 'label_list.json']:
        checks[str(p.relative_to(out_root))] = sha256(p)
    (out_root / f'PROVENANCE_sroie_repaired_{today}.md').write_text(
        '# SROIE repaired-alignment bundle\n\n'
        f'Built {datetime.date.today().isoformat()} from '
        f'`{args.root}` (official ICDAR-2019 Task-3 OCR + entities).\n\n'
        '## Why this lineage exists\n\n'
        'The pre-registered strict gate reaches 94.45% because Task-3 field '
        'strings disagree with the OCR transcription on punctuation (mostly '
        'multi-line ADDRESS) and on currency/decimal formatting (TOTAL). '
        'Relaxing exactly those two conventions clears the 95% gate. The '
        'original exclusion decision in results/sroie_gate_report_20260705.md '
        'is unchanged and still reported.\n\n'
        '## Alignment\n\n'
        + json.dumps({k: {'alignment_rate': v['alignment_rate'],
                          'docs': v['docs'], 'tokens': v['tokens'],
                          'o_fraction': v['o_fraction']}
                      for k, v in split_stats.items()}, indent=2)
        + '\n\nCounts by winning matcher: '
        + json.dumps(dict(matcher_totals), indent=2)
        + '\n\n## Checksums\n\n'
        + json.dumps(checks, indent=2)
        + '\n\n## Caveats\n\n'
        '- Non-contiguous multi-line matching was NOT used (recovers only '
        '~0.1% and can stitch unrelated tokens).\n'
        '- Fields whose spans would overlap an already-claimed span are '
        'dropped rather than silently overwritten.\n'
        '- Residual ~2.5% of field instances remain unmatched: genuine OCR '
        'divergence or GT typos, not repairable by alignment.\n'
        '- PENDING PI SIGN-OFF before any number enters the paper.\n')
    print(f'\nWrote bundle to {out_root}')
    print(f'matchers: {dict(matcher_totals)}')


if __name__ == '__main__':
    main()
