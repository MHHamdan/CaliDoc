#!/usr/bin/env python3
"""
E7 step 1 -- diagnose WHY the strict SROIE alignment gate fails.

The pre-registered gate (scripts/data_prep/sroie_prep_gate.py) requires each
Task-3 field string to match a CONTIGUOUS run of OCR tokens exactly, after
whitespace-stripping and casefolding only. It scores 94.68% (train) / 94.02%
(test), just under the 95% threshold, which is why SROIE was excluded.

Before attempting a repair we need to know what the residual ~5-6% actually
is. This script re-runs the alignment and classifies every failure into a
cause, escalating through progressively weaker matchers and recording which
one first succeeds:

  strict          the gate's own matcher (whitespace + casefold)
  punctuation     additionally ignore all non-alphanumeric characters
  numeric         TOTAL-style: compare as a number after stripping currency
                  symbols/letters, tolerating 9 vs 9.0 vs 9.00
  noncontiguous   the field's tokens all appear in order but with unrelated
                  tokens interleaved (the multi-line address signature)
  unmatched       no token subsequence yields the field -- genuine OCR
                  divergence, not repairable by alignment

A field that is only recoverable at the 'unmatched' level is a hard ceiling on
any realignment effort; the mix of causes decides whether E7 can work.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sroie_prep_gate import find_span, norm_chars, parse_ocr_file  # noqa: E402

FIELDS = ('company', 'date', 'address', 'total')


def norm_alnum(s):
    return re.sub(r'[^0-9a-z]', '', s.casefold())


def as_number(s):
    m = re.findall(r'\d+(?:[.,]\d+)?', s.replace(' ', ''))
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


def numeric_match(tokens, entity):
    want = as_number(entity)
    if want is None:
        return None
    for i, tok in enumerate(tokens):
        if as_number(tok) == want:
            return (i, i)
    return None


def noncontiguous_match(tokens_alnum, entity):
    """All of the field's characters appear across tokens in order, but the
    tokens are not adjacent. Signature of multi-line addresses."""
    target = norm_alnum(entity)
    if not target:
        return None
    acc, used = 0, []
    for i, tok in enumerate(tokens_alnum):
        if not tok:
            continue
        if target[acc:acc + len(tok)] == tok:
            acc += len(tok)
            used.append(i)
            if acc == len(target):
                return (used[0], used[-1])
    return None


def diagnose(split_dir_ocr, split_dir_ent, label):
    causes = Counter()
    per_field = defaultdict(Counter)
    n_fields = 0
    examples = defaultdict(list)

    ent_files = sorted(split_dir_ent.glob('*.txt'))
    for ef in ent_files:
        ocr = split_dir_ocr / ef.name
        if not ocr.exists():
            continue
        import json
        try:
            ents = json.loads(ef.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            continue
        tokens, _ = parse_ocr_file(ocr)
        tnorm = [norm_chars(t) for t in tokens]
        talnum = [norm_alnum(t) for t in tokens]

        for field in FIELDS:
            val = ents.get(field) or ents.get(field.upper())
            if not val:
                continue
            n_fields += 1
            if find_span(tnorm, val) is not None:
                causes['strict'] += 1; per_field[field]['strict'] += 1
            elif find_span_alnum(talnum, val) is not None:
                causes['punctuation'] += 1; per_field[field]['punctuation'] += 1
                if len(examples['punctuation']) < 5:
                    examples['punctuation'].append((ef.stem, field, val[:60]))
            elif field == 'total' and numeric_match(tokens, val) is not None:
                causes['numeric'] += 1; per_field[field]['numeric'] += 1
                if len(examples['numeric']) < 5:
                    examples['numeric'].append((ef.stem, field, val[:60]))
            elif noncontiguous_match(talnum, val) is not None:
                causes['noncontiguous'] += 1; per_field[field]['noncontiguous'] += 1
                if len(examples['noncontiguous']) < 5:
                    examples['noncontiguous'].append((ef.stem, field, val[:60]))
            else:
                causes['unmatched'] += 1; per_field[field]['unmatched'] += 1
                if len(examples['unmatched']) < 8:
                    examples['unmatched'].append((ef.stem, field, val[:60]))

    print(f'\n===== {label}: {n_fields} field instances over {len(ent_files)} docs =====')
    cum = 0
    for cause in ('strict', 'punctuation', 'numeric', 'noncontiguous', 'unmatched'):
        c = causes[cause]
        cum += c
        print(f'  {cause:<14} {c:6d}  {100*c/n_fields:6.2f}%   cumulative {100*cum/n_fields:6.2f}%')
    print('  by field:')
    for field in FIELDS:
        tot = sum(per_field[field].values())
        if not tot:
            continue
        strict = per_field[field]['strict']
        rec = tot - per_field[field]['unmatched']
        print(f'    {field:<9} n={tot:5d}  strict {100*strict/tot:6.2f}%  '
              f'recoverable {100*rec/tot:6.2f}%  hard-fail {per_field[field]["unmatched"]:4d}')
    for k in ('punctuation', 'numeric', 'noncontiguous', 'unmatched'):
        if examples[k]:
            print(f'  examples [{k}]:')
            for doc, f, v in examples[k]:
                print(f'    {doc} {f}: {v!r}')
    return causes, n_fields


def main():
    root = Path('data/SROIE_official/SROIE2019')
    total = Counter(); n_all = 0
    for split in ('train', 'test'):
        ocr_dir = root / split / 'box'
        ent_dir = root / split / 'entities'
        if not ocr_dir.exists() or not ent_dir.exists():
            print(f'missing {ocr_dir} or {ent_dir}')
            continue
        c, n = diagnose(ocr_dir, ent_dir, split)
        total.update(c); n_all += n

    if n_all:
        print(f'\n===== combined: {n_all} field instances =====')
        cum = 0
        for cause in ('strict', 'punctuation', 'numeric', 'noncontiguous', 'unmatched'):
            cum += total[cause]
            print(f'  {cause:<14} {total[cause]:6d}  {100*total[cause]/n_all:6.2f}%'
                  f'   cumulative {100*cum/n_all:6.2f}%')


if __name__ == '__main__':
    main()
