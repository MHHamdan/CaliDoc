#!/usr/bin/env python3
"""
Cross-check every number the manuscripts assert against the result JSONs.

This project has repeatedly shipped stale numbers through edits (the CORD 95.92
figure survived a full rewrite), so the defence is mechanical rather than
careful reading: each check names a value, computes it from the JSON that owns
it, and asserts the manuscripts contain that value and NOT any superseded one.

Exit code is non-zero if any check fails, so it can gate a build.

Usage:
  python scripts/validate_paper_numbers.py --results_dir results/local_20260803
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

TEX = ['docs/paper/calidoc_paper.tex',
       'docs/paper/calidoc_eccv.tex',
       'docs/paper/sec5_practical_lessons.tex',
       'docs/paper/supplementary.tex',
       'docs/paper/supplementary_eccv.tex']

# Numbers that were true once and must never reappear.
SUPERSEDED = {
    '95.92': 'legacy CORD F1 from the corrupted 61-label mapping',
    '83.80': 'legacy FUNSD F1 (pre es_holdout protocol)',
    '2.06': 'legacy CORD ECE',
}

# Claims from the superseded cluster lineage. These are prose, not numbers, and
# were found by reading rather than by tooling -- which is exactly why they are
# encoded here now. A bare mention is allowed only where the text explicitly
# frames it as the earlier lineage.
STALE_CLAIMS = {
    'H100': 'cluster hardware; local runs use RTX PRO 6000 Blackwell',
    'PyTorch 2.12': 'cluster stack; local runs use PyTorch 2.9.1',
    'CUDA 12.2': 'cluster stack; local runs use CUDA 12.8',
}
STALE_CLAIM_ALLOWED_CONTEXT = [
    'earlier lineage', 'not the source', 'superseded', 'previously',
]
# A superseded value may appear ONLY inside an explicit callout that marks it
# as no longer valid. Anything else is a stale number that survived an edit.
SUPERSEDED_ALLOWED_CONTEXT = [
    'superseded', 'previously reported', 'legacy',
    'now treated as invalid', 'no longer', 'incorrect',
]


def load(results_dir, name):
    p = Path(results_dir) / name
    return json.loads(p.read_text()) if p.exists() else None


def tex_sources():
    out = {}
    for t in TEX:
        p = Path(t)
        if p.exists():
            out[t] = p.read_text()
    return out


def check(results, label, value, sources, tol_str=None):
    """Assert `value` (formatted string) appears in at least one manuscript."""
    needle = tol_str or value
    hits = [name for name, s in sources.items() if needle in s]
    results.append({
        'check': label, 'value': needle,
        'status': 'PASS' if hits else 'MISSING',
        'found_in': hits,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results/local_20260803')
    ap.add_argument('--strict', action='store_true',
                    help='fail on MISSING as well as on superseded numbers')
    args = ap.parse_args()
    rd = args.results_dir
    sources = tex_sources()
    if not sources:
        print('no manuscripts found'); return 1
    results = []

    # ---- SROIE audit numbers (the corrected marginal-vs-Mondrian story) ----
    sro = load(rd, 'phase2_sroie.json')
    if sro:
        r = sro['results']; ps = r['per_seed']; a = 'alpha_0.1'
        marg = float(np.mean([ps[s]['lac'][a]['coverage'] for s in ps])) * 100
        check(results, 'SROIE marginal coverage @90', f'{marg:.1f}', sources)

        agg, mond = {}, {}
        for s in ps:
            e = ps[s]['lac'][a]
            for k, v in e['per_class_coverage'].items():
                agg.setdefault(k, []).append(v['coverage'])
            for k, v in e['mondrian']['per_class'].items():
                if 'qhat' in v:
                    mond.setdefault(k, []).append(v['qhat'])
        if 'B-ADDRESS' in mond:
            q = float(np.mean(mond['B-ADDRESS']))
            check(results, 'SROIE Mondrian qhat B-ADDRESS', f'{q:.4f}', sources)
        if 'O' in mond:
            q = float(np.mean(mond['O']))
            check(results, 'SROIE Mondrian qhat O', f'{q:.4f}', sources)

        # Spread is max/min over EVERY evaluated class, not a hardcoded pair.
        # Hardcoding B-ADDRESS/O certified a spread that excluded I-DATE and
        # I-TOTAL, both of which clear the feasibility bound and both of which
        # carry larger quantiles than B-ADDRESS.
        if mond:
            means = {k: float(np.mean(v)) for k, v in mond.items() if np.mean(v) > 0}
            if len(means) >= 2:
                hi_k, hi = max(means.items(), key=lambda kv: kv[1])
                lo_k, lo = min(means.items(), key=lambda kv: kv[1])
                spread = hi / lo
                results.append({
                    'check': 'SROIE quantile spread factor (max/min over all classes)',
                    'value': f'{spread:.0f}x  [{hi_k} {hi:.4f} / {lo_k} {lo:.4f}; '
                             f'{len(means)} classes]',
                    'status': 'PASS' if f'factor of ${spread:.0f}$' in
                              ''.join(sources.values()) else 'MISSING',
                    'found_in': [],
                })
                # The manuscript must not print a spread computed on a subset.
                subset = float(np.mean(mond['B-ADDRESS'])) / q if 'B-ADDRESS' in mond and q else None
                if subset and f'factor of ${subset:.0f}$' in ''.join(sources.values()):
                    results.append({
                        'check': 'SROIE spread computed on a FILTERED class list',
                        'value': f'manuscript prints {subset:.0f}x (B-ADDRESS/O) '
                                 f'but max/min over all {len(means)} classes is {spread:.0f}x',
                        'status': 'FAIL',
                        'found_in': [],
                    })
        # every B- class must be at 0.0 under the marginal threshold
        zeros = [k for k, v in agg.items()
                 if k.startswith('B-') and float(np.mean(v)) == 0.0]
        results.append({
            'check': 'SROIE B- classes at 0% under marginal threshold',
            'value': f'{len(zeros)} classes: {sorted(zeros)}',
            'status': 'PASS' if len(zeros) == 4 else 'REVIEW',
            'found_in': [],
        })

    # ---- per-dataset headline metrics -------------------------------------
    ms = load(rd, 'multiseed_summary.json')
    if ms:
        for ds, d in ms['datasets'].items():
            agg = d['aggregate']
            f1 = agg['f1']['mean']; sd = agg['f1']['std']
            n = agg['f1']['n_seeds']
            results.append({
                'check': f'{ds} test F1 ({n} seeds)',
                'value': f'{f1:.2f} +- {sd:.2f}',
                'status': 'PASS' if f'{f1:.2f}' in ''.join(sources.values())
                          else 'MISSING',
                'found_in': [],
            })
    else:
        for ds in ('cord', 'funsd', 'wildreceipt'):
            d = load(rd, f'phase2_{ds}.json')
            if d:
                f1 = d['results']['test_metrics']['f1'] * 100
                results.append({
                    'check': f'{ds} test F1 (seed 42 only)',
                    'value': f'{f1:.2f}',
                    'status': 'INFO (multiseed summary not yet written)',
                    'found_in': [],
                })

    # ---- superseded numbers must not reappear ------------------------------
    bad = []
    for num, why in SUPERSEDED.items():
        for name, s in sources.items():
            for m in re.finditer(re.escape(num), s):
                window = s[max(0, m.start() - 260):m.start() + 160].lower()
                if any(c in window for c in SUPERSEDED_ALLOWED_CONTEXT):
                    continue
                bad.append((name, num, why))
    for name, num, why in bad:
        results.append({'check': f'superseded {num} ({why})',
                        'value': num, 'status': 'FAIL', 'found_in': [name]})

    # ---- single-seed cluster values still embedded in prose ----------------
    # The results TABLE is generated and already multi-seed, but the abstract,
    # contributions and discussion still quote the old single-run figures.
    # Flagged as REVIEW (not FAIL) while seeds are still landing; they must be
    # replaced by the multi-seed means before submission.
    single_seed = {'95.35': 'CORD', '81.82': 'FUNSD', '94.16': 'WildReceipt',
                   '0.76': 'Phase-1 perturbation mean F1 drop',
                   '2.11': 'gap-to-SOTA under the segment-level protocol'}
    for num, ds in single_seed.items():
        hits = []
        for n, s in sources.items():
            for m in re.finditer(re.escape(num), s):
                # Skip matches inside identifiers (arXiv ids, DOIs, years) and
                # inside longer numbers -- "2.11" occurs in "arXiv:2511.22521".
                before = s[m.start() - 1] if m.start() else ' '
                after = s[m.end()] if m.end() < len(s) else ' '
                if before.isdigit() or after.isdigit():
                    continue
                window = s[max(0, m.start() - 120):m.end() + 40].lower()
                if 'arxiv' in window or 'doi' in window:
                    continue
                hits.append(Path(n).name)
                break
        if hits:
            results.append({
                'check': f'single-seed {ds} value still in prose',
                'value': num, 'status': 'REVIEW', 'found_in': sorted(set(hits))})

    # ---- stale prose claims from the cluster lineage -----------------------
    for claim, why in STALE_CLAIMS.items():
        for name, s in sources.items():
            for m in re.finditer(re.escape(claim), s):
                window = s[max(0, m.start() - 400):m.start() + 260].lower()
                if any(c in window for c in STALE_CLAIM_ALLOWED_CONTEXT):
                    continue
                results.append({'check': f'stale claim "{claim}" ({why})',
                                'value': claim, 'status': 'FAIL',
                                'found_in': [name]})
                break

    width = max(len(r['check']) for r in results) if results else 10
    n_fail = n_missing = 0
    for r in results:
        st = r['status']
        if st == 'FAIL':
            n_fail += 1
        elif st == 'MISSING':
            n_missing += 1
        loc = (' [' + ', '.join(Path(f).name for f in r['found_in']) + ']'
               if r['found_in'] else '')
        print(f'{st:<8} {r["check"]:<{width}}  {r["value"]}{loc}')

    print(f'\n{len(results)} checks: {n_fail} FAIL, {n_missing} MISSING')
    if n_fail:
        return 1
    if n_missing and args.strict:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
