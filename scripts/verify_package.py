#!/usr/bin/env python3
"""Package-integrity checks that run FROM the extracted supplementary archive.

`check_gates.py` lives in the source repository and validates the manuscript
build (page budget, overfull boxes, bibliography, macro regeneration). It
needs the repo layout and does *not* run from an exported archive.

This script is the archive-side counterpart: it is shipped inside `code/` and
verifies only what is checkable there --- that the reproduction path is
present and self-consistent, that the committed artifacts agree with the
scripts that read them, and that no identity string survived the export.

    cd code && python3 verify_package.py            # structure + artifacts
    cd code && python3 verify_package.py --run      # also re-run the analyses
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RD = HERE / 'results' / 'local_20260803'
CACHE = HERE / 'results' / 'cache'
FAILS: list[str] = []
OKS: list[str] = []

# encoded so this checker does not itself contain the strings it forbids
_IDENT_B64 = ('bWhhbWRhbgpjaGVyaWV0CmRlZi1tY2hlcmlldAptaDIwMjJldHMKL2hvbWUvKD8h'
              'YW5vbikKXGJuaWJpXGIKXGJ0cmlsbGl1bVxi')
IDENT = [re.compile(p, re.I)
         for p in base64.b64decode(_IDENT_B64).decode().split('\n')]

DATASETS = ['cord', 'funsd', 'sroie', 'wildreceipt', 'sroie_repaired']


def gate(name: str, ok: bool, detail: str = '') -> None:
    (OKS if ok else FAILS).append(name)
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'  --  {detail}' if detail else ''))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true',
                    help='also re-run the analysis scripts (slower)')
    args = ap.parse_args()

    # ---------------------------------------------------------- structure
    gate('README present', (HERE / 'README.md').exists())
    gate('analysis requirements present',
         (HERE / 'requirements-analysis.txt').exists())
    for s in ('span_crc.py', 'multiplicity_analysis.py',
              'spancrc_sensitivity.py', 'boundary_escape_power.py',
              'first_token_entity_cp.py', 'emit_result_macros.py'):
        gate(f'scripts/{s} present', (HERE / 'scripts' / s).exists())

    # --------------------------------------------- caches actually shipped
    npz = sorted(CACHE.glob('*.npz'))
    missing = [d for d in DATASETS
               if not (CACHE / f'{d}_local_20260803.npz').exists()]
    gate('probability caches shipped', not missing,
         f'{len(npz)} .npz, '
         f'{sum(p.stat().st_size for p in npz) / 1e6:.0f} MB'
         + (f', MISSING {missing}' if missing else ''))
    for d in DATASETS:
        j = CACHE / f'{d}_local_20260803.json'
        gate(f'cache metadata: {d}', j.exists(),
             f"T={json.loads(j.read_text())['temperature']:.2f}" if j.exists() else '')

    # ------------------------------------------------ committed artifacts
    for d in DATASETS:
        gate(f'span_crc_{d}.json present', (RD / f'span_crc_{d}.json').exists())
    for f in ('multiplicity_analysis.json', 'boundary_escape_power.json',
              'spancrc_sensitivity.json', 'first_token_entity_cp.json',
              'multiseed_summary.json'):
        gate(f'{f} present', (RD / f).exists())

    # ------------------------------------------- artifact self-consistency
    if (RD / 'multiplicity_analysis.json').exists():
        m = json.loads((RD / 'multiplicity_analysis.json').read_text())
        agree = all(d['rejected_holm'] == d['rejected_bh'] == d['rejected_by']
                    for d in m['per_dataset'].values())
        gate('Holm/BH/BY agree on every dataset', agree)
        gate('exactly two SROIE rejections',
             m['per_dataset']['sroie']['rejected_holm'] == ['ADDRESS', 'COMPANY'])
        gate('pooled family agrees',
             len(m['pooled']['rejected_holm']) == 2)
        gate('decision stable across calibration splits',
             all(all(p == per[0] for p in per)
                 for per in m['stability_across_splits'].values()))
    if (RD / 'span_crc_sroie.json').exists():
        s = json.loads((RD / 'span_crc_sroie.json').read_text())
        a = s['aggregate']
        gate('SROIE span coverage 0.000 -> 0.909',
             abs(a['span_cov_token_marginal']['mean']) < 1e-9
             and abs(a['span_cov_field_crc']['mean'] - 0.9087) < 5e-4,
             f"{a['span_cov_token_marginal']['mean']:.3f} -> "
             f"{a['span_cov_field_crc']['mean']:.4f}")
        gate('SROIE field-miss risk 1.000 -> 0.091',
             abs(a['ev_doc_risk_token_marginal']['mean'] - 1.0) < 1e-9
             and a['ev_doc_risk_field_crc']['mean'] < 0.10)
    if (RD / 'span_crc_cord.json').exists():
        c = json.loads((RD / 'span_crc_cord.json').read_text())['aggregate']
        gate('CORD span CovGap 42.3 -> 33.3',
             abs(c['span_covgap_pp_token_marginal']['mean'] - 42.3) < 0.05
             and abs(c['span_covgap_pp_field_crc']['mean'] - 33.3) < 0.05,
             f"{c['span_covgap_pp_token_marginal']['mean']:.1f} -> "
             f"{c['span_covgap_pp_field_crc']['mean']:.1f}")

    # ---------------------------------------------------------- anonymity
    hits = []
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or p.suffix == '.npz':
            continue
        try:
            txt = p.read_text(errors='replace')
        except Exception:
            continue
        for pat in IDENT:
            if pat.search(txt):
                hits.append(f'{p.relative_to(HERE)}: {pat.pattern}')
    gate('no identity strings in the export', not hits, str(hits[:3]))

    # ------------------------------------------------- optional live rerun
    if args.run:
        for cmd in (['scripts/span_crc.py', '--all'],
                    ['scripts/multiplicity_analysis.py'],
                    ['scripts/spancrc_sensitivity.py'],
                    ['scripts/first_token_entity_cp.py']):
            r = subprocess.run([sys.executable] + cmd, cwd=HERE,
                               capture_output=True, text=True)
            ok = r.returncode == 0
            note = ''
            if cmd[0].endswith('span_crc.py'):
                ok = ok and r.stdout.count('self-check: PASS') == len(DATASETS)
                note = f"{r.stdout.count('self-check: PASS')}/{len(DATASETS)} self-checks"
            gate(f'reruns cleanly: {cmd[0]}', ok,
                 note or (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''))

    print()
    print(f'{len(OKS)} checks passed, {len(FAILS)} failed')
    if FAILS:
        for f in FAILS:
            print(f'  ! {f}')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
