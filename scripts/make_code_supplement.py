#!/usr/bin/env python3
"""Build the anonymized code supplement ZIP for the WACV 2027 submission.

Exports a curated subset of the repository -- the scripts needed to reproduce
the central results, the committed result artifacts the paper's tables and
figures read, and environment files -- rather than zipping the working tree.
Nothing is copied without passing an identity scan first.

Sanitization:
  * cluster/account identifiers (`def-mcheriet`), home paths (`/home/<user>`),
    the author's username, institutional names, and author-revealing URLs are
    rewritten to neutral placeholders;
  * the entire `slurm/` tree and `scripts/slurm/`, `scripts/nibi_*` are
    excluded (cluster-specific and identifying, and not needed to reproduce
    anything in the paper -- every reported analysis runs post-hoc on the
    cached probabilities included here);
  * no git history, no `.git`, no notebooks with execution metadata.

After staging, a recursive scan re-checks every exported file (including
inside JSON) for the forbidden patterns and aborts on any hit.

    python3 scripts/make_code_supplement.py            # build + scan
    python3 scripts/make_code_supplement.py --scan_only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGE = REPO / 'build' / 'code_supplement'
ZIP = REPO / 'docs' / 'paper-wacv2027' / 'calidoc_code_supplement.zip'
FINAL = REPO / 'docs' / 'paper-wacv2027' / 'calidoc_wacv2027_supplementary.zip'

# Files copied verbatim (after sanitization). Reproduce every number in the
# paper's tables and figures from the cached probabilities.
SCRIPTS = [
    'scripts/span_crc.py',
    'scripts/spancrc_sensitivity.py',
    'scripts/boundary_escape_power.py',
    'scripts/first_token_entity_cp.py',
    'scripts/conditional_cp_baselines.py',
    'scripts/label_error_baselines.py',
    'scripts/noise_dose_response.py',
    'scripts/aggregate_noise_dose.py',
    'scripts/evaluate_conformal_phase2.py',
    'scripts/evaluate_cross_conformal.py',
    'scripts/evaluate_mc_dropout.py',
    'scripts/evaluate_raps.py',
    'scripts/evaluate_coverage_under_perturbations.py',
    'scripts/ablations_phase2.py',
    'scripts/cache_probs.py',
    'scripts/aggregate_multiseed.py',
    'scripts/emit_main_results_table.py',
    'scripts/emit_supplementary_tables.py',
    'scripts/emit_label_error_table.py',
    'scripts/emit_wacv_supplementary.py',
    'scripts/create_spancrc_figures.py',
    'scripts/create_audit_figures.py',
    'scripts/data_prep/sroie_prep_gate.py',
    'scripts/data_prep/sroie_alignment_diagnosis.py',
    'scripts/multiplicity_analysis.py',
    'scripts/emit_result_macros.py',
    'requirements.txt',
    'requirements-analysis.txt',
]
RESULT_GLOBS = [
    'results/local_20260803/span_crc_*.json',
    'results/local_20260803/spancrc_sensitivity.json',
    'results/local_20260803/boundary_escape_power.json',
    'results/local_20260803/first_token_entity_cp.json',
    'results/local_20260803/conditional_cp_*.json',
    'results/local_20260803/ablations_*.json',
    'results/local_20260803/phase2_*.json',
    'results/local_20260803/multiseed_summary.json',
    'results/local_20260803/noise_dose_summary.json',
    'results/local_20260803/label_error_baselines_*.json',
    'results/local_20260803/perturbation_coverage_*.json',
    'results/local_20260803/multiplicity_analysis.json',
    'results/local_20260803/boundary_escape_power.json',
    'results/local_20260803/spancrc_sensitivity.json',
    'results/local_20260803/first_token_entity_cp.json',
    'results/sroie_gate_report_*.md',
    'results/sroie_alignment_diagnosis_*.txt',
]
# The .npz probability caches ARE shipped (~26 MB): without them the
# advertised post-hoc reproduction path does not run at all.
CACHE_GLOBS = ['results/cache/*.json', 'results/cache/*.npz']

# (pattern, replacement) -- applied to every text file that is exported.
SANITIZE = [
    (re.compile(r'def-mcheriet'), 'ANON-ACCOUNT'),
    (re.compile(r'/home/mhamdan'), '/home/anon'),
    (re.compile(r'\bmhamdan\b'), 'anon'),
    (re.compile(r'mh2022ets@gmail\.com'), 'anonymous@example.invalid'),
    (re.compile(r'https?://github\.com/[^\s"\')]+'), 'ANONYMIZED-URL'),
    (re.compile(r'\bCheriet\b', re.I), 'ANON'),
    (re.compile(r'\b(ETS|Ecole de technologie superieure|'
                r'\xc9cole de technologie sup\xe9rieure)\b'), 'ANON-INSTITUTION'),
    (re.compile(r'\b(nibi|trillium|narval|beluga|cedar|graham)\b', re.I),
     'ANON-CLUSTER'),
]
# Patterns that must not survive anywhere in the export.
FORBIDDEN = [
    re.compile(r'def-mcheriet', re.I),
    re.compile(r'mhamdan', re.I),
    re.compile(r'cheriet', re.I),
    re.compile(r'/home/(?!anon)[a-z0-9_.-]+', re.I),
    re.compile(r'github\.com/', re.I),
    re.compile(r'mh2022ets', re.I),
    re.compile(r'\b(nibi|trillium)\b', re.I),
]
TEXT_SUFFIXES = {'.py', '.sh', '.md', '.txt', '.json', '.yaml', '.yml', '.cfg'}

README = """# CaliDoc -- anonymized code supplement

Reproduces every number in the paper's tables and figures. All analyses in the
paper run *post-hoc* on cached, temperature-scaled probabilities; no GPU and no
retraining are required to regenerate the results below.

## Layout

    scripts/                analysis + table/figure emitters
    results/local_20260803/ committed result artifacts read by the tables
    results/cache/*.json    cache metadata (label maps, temperatures, splits)
    requirements.txt        environment

The temperature-scaled probability caches (`results/cache/*.npz`) are
**included**, so the commands below run as-is with no GPU, no checkpoint and
no download. `scripts/cache_probs.py` documents how they were produced from a
fine-tuned checkpoint, if you want to regenerate them.

## Install

    pip install -r requirements-analysis.txt

That is numpy + scipy + matplotlib. `requirements.txt` is the full training
environment and is only needed to re-train a backbone or rebuild the caches.

## Reproducing the central results

    python3 scripts/span_crc.py --all           # SpanCRC + boundary-escape
    python3 scripts/multiplicity_analysis.py    # Holm / BH / BY + blocked null
    python3 scripts/spancrc_sensitivity.py      # m0 sweep, candidate-rule sizes
    python3 scripts/boundary_escape_power.py    # size + power validation
    python3 scripts/first_token_entity_cp.py    # first-token comparison
    python3 scripts/emit_result_macros.py       # all headline numbers as macros
    python3 scripts/emit_wacv_supplementary.py  # supplementary tables

Runtime is a few minutes in total on a laptop CPU; `boundary_escape_power.py`
is the slowest (~10 min: 600 synthetic-null replicates plus 30 corruption
seeds x 5 doses x 7 fields x 2 variants).

`span_crc.py` refuses to write its output unless it reproduces two
independently committed artifacts to 1e-4; `spancrc_sensitivity.py` asserts
that its m0=25 row equals the committed SpanCRC result.

## Checking this package

    python3 verify_package.py          # structure, artifacts, anonymity
    python3 verify_package.py --run    # also re-runs the analyses

(The manuscript-side gates -- page budget, overfull boxes, bibliography,
macro regeneration -- run in the authors' source repository at build time and
are not reproducible from this archive, which contains no LaTeX sources.)

## Data

FUNSD, CORD, WildReceipt and SROIE are public. The SROIE alignment gate and
its diagnosis (`scripts/data_prep/sroie_prep_gate.py`,
`sroie_alignment_diagnosis.py`) run against the official ICDAR-2019 Task-3
release; `results/sroie_gate_report_*.md` records the gate outcome and the
three independent data lineages it was verified on.
"""


def sanitize_text(s: str) -> str:
    for pat, rep in SANITIZE:
        s = pat.sub(rep, s)
    return s


def stage() -> list[Path]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    copied = []
    paths = [REPO / p for p in SCRIPTS]
    for g in RESULT_GLOBS + CACHE_GLOBS:
        paths.extend(sorted(REPO.glob(g)))
    for src in paths:
        if not src.exists():
            print(f'  skip (missing): {src.relative_to(REPO)}')
            continue
        dst = STAGE / src.relative_to(REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in TEXT_SUFFIXES:
            dst.write_text(sanitize_text(src.read_text(errors='replace')))
        else:
            shutil.copy2(src, dst)
        copied.append(dst)
    (STAGE / 'README.md').write_text(README)
    copied.append(STAGE / 'README.md')
    vp = REPO / 'scripts' / 'verify_package.py'
    (STAGE / 'verify_package.py').write_text(sanitize_text(vp.read_text()))
    copied.append(STAGE / 'verify_package.py')
    return copied


def scan(root: Path) -> list[str]:
    hits = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors='replace')
        except Exception:
            continue
        for pat in FORBIDDEN:
            for m in pat.finditer(text):
                line = text[:m.start()].count('\n') + 1
                hits.append(f'{p.relative_to(root)}:{line}: {m.group(0)!r}')
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan_only', action='store_true')
    args = ap.parse_args()

    if not args.scan_only:
        files = stage()
        print(f'staged {len(files)} files under {STAGE}')

    hits = scan(STAGE)
    if hits:
        print(f'IDENTITY SCAN FAILED ({len(hits)} hits):')
        for h in hits[:40]:
            print(f'  ! {h}')
        raise SystemExit(1)
    print('identity scan: PASS (0 hits across '
          f'{sum(1 for _ in STAGE.rglob("*") if _.is_file())} files)')

    if args.scan_only:
        return
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():
                # deterministic timestamps: no filesystem metadata leaks
                info = zipfile.ZipInfo(str(p.relative_to(STAGE)),
                                       date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes())
    print(f'wrote {ZIP} ({ZIP.stat().st_size/1024:.0f} KB)')

    final = build_final()
    hits = scan_zip(final)
    if hits:
        print(f'FINAL ARCHIVE IDENTITY SCAN FAILED ({len(hits)} hits):')
        for h in hits[:20]:
            print(f'  ! {h}')
        raise SystemExit(1)
    with zipfile.ZipFile(final) as z:
        n = len(z.namelist())
    print(f'wrote {final} ({final.stat().st_size/1024/1024:.1f} MB, {n} entries)')
    print(f'final-archive identity scan: PASS')


def build_final(readme_extra=''):
    """Assemble the single upload-ready supplementary archive."""
    pdf = REPO / 'docs' / 'paper-wacv2027' / 'supplementary.pdf'
    if not pdf.exists():
        raise SystemExit('supplementary.pdf missing -- run ./build_supp.sh')
    readme = f"""CaliDoc -- WACV 2027 supplementary material
===========================================

CONTENTS

  supplementary.pdf   Supplementary document (sections S1-S11): full
                      protocol and reproducibility details, complete backbone
                      and conformal results, SpanCRC derivation and per-field
                      detail, the boundary-escape null, multiplicity and
                      power analyses, the SROIE audit, label-noise baselines,
                      calibration, robustness grids, and the reproducibility
                      checklist.

  code/               Anonymized code export, INCLUDING the temperature-scaled
                      probability caches. Reproduces every number in the
                      paper's tables and figures post-hoc -- no GPU, no
                      checkpoint, no download. See code/README.md.

REPRODUCTION IN BRIEF

  cd code
  pip install -r requirements-analysis.txt      # numpy, scipy, matplotlib
  python3 scripts/span_crc.py --all             # SpanCRC + boundary-escape
  python3 scripts/multiplicity_analysis.py      # Holm / BH / BY + blocked null
  python3 scripts/spancrc_sensitivity.py        # m0 sweep
  python3 scripts/boundary_escape_power.py      # size + power (~10 min)
  python3 scripts/first_token_entity_cp.py      # first-token comparison

  The analysis scripts self-check against independently committed artifacts
  and refuse to write output on a mismatch.

  python3 verify_package.py                     # package-integrity checks

NOTES

  * requirements.txt is the full training environment; it is NOT needed for
    the reproduction path above.
  * All datasets used (FUNSD, CORD, WildReceipt, SROIE) are public.
  * This archive is anonymized: no author names, affiliations, usernames,
    cluster accounts, home paths or author-identifying URLs. An automated
    recursive scan is run over this exact archive before release.
{readme_extra}"""
    if FINAL.exists():
        FINAL.unlink()
    with zipfile.ZipFile(FINAL, 'w', zipfile.ZIP_DEFLATED) as z:
        def add(arc, data):
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
        add('README.txt', readme)
        add('supplementary.pdf', pdf.read_bytes())
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():
                arc = f'code/{p.relative_to(STAGE)}'
                # keep the build/reproduce scripts executable on extraction
                mode = 0o755 if p.suffix == '.sh' else 0o644
                info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = mode << 16
                z.writestr(info, p.read_bytes())
    return FINAL


def scan_zip(path):
    """Recursive identity scan of a finished archive."""
    hits = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            try:
                text = z.read(n).decode('utf-8', errors='replace')
            except Exception:
                continue
            for pat in FORBIDDEN:
                for m in pat.finditer(text):
                    hits.append(f'{n}: {m.group(0)!r}')
    return hits


if __name__ == '__main__':
    main()
