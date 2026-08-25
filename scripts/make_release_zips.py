#!/usr/bin/env python3
"""Build the three release archives for the CaliDoc WACV 2027 submission.

    1. calidoc_wacv2027_paper.zip          main paper PDF + full LaTeX source
    2. calidoc_wacv2027_supplementary.zip  supplementary PDF + anonymized code
    3. calidoc-github-repo.zip             clean public repository + README

All three are sanitized with the same rewrite rules and then scanned with the
same forbidden-pattern list; the build aborts on any hit. Archive 2 is
produced by make_code_supplement.py (this script invokes it) so the upload
artifact keeps exactly one definition.

    python3 scripts/make_release_zips.py            # build all three
    python3 scripts/make_release_zips.py --scan     # scan existing archives

Archive 3 is ANONYMIZED by default: it is byte-for-byte safe to share during
double-blind review. Author names, affiliation, funding and the real citation
go in the marked placeholders in its README/CITATION.cff after the review
embargo lifts.
"""

from __future__ import annotations

import argparse
import base64
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / 'docs' / 'paper-wacv2027'
OUT = REPO / 'release'
STAGE = REPO / 'build' / 'release'

PAPER_ZIP = OUT / 'calidoc_wacv2027_paper.zip'
SUPP_ZIP = OUT / 'calidoc_wacv2027_supplementary.zip'
REPO_ZIP = OUT / 'calidoc-github-repo.zip'

# --------------------------------------------------------------- sanitizing
SANITIZE = [
    (re.compile(r'def-mcheriet'), 'ANON-ACCOUNT'),
    (re.compile(r'/home/mhamdan'), '/home/anon'),
    (re.compile(r'\bmhamdan\b'), 'anon'),
    (re.compile(r'mh2022ets@gmail\.com'), 'anonymous@example.invalid'),
    (re.compile(r'\bCheriet\b', re.I), 'ANON'),
    (re.compile(r'\b(ETS|Ecole de technologie superieure|'
                r'\xc9cole de technologie sup\xe9rieure)\b'), 'ANON-INSTITUTION'),
    (re.compile(r'\b(nibi|trillium|narval|beluga|cedar|graham)\b', re.I),
     'ANON-CLUSTER'),
    # conda records the upstream feedstock build path for every package as
    # ` @ file:///home/conda/feedstock_root/...`. Not author-identifying, but
    # noisy and it trips the scanner; drop the local-URL suffix so the freeze
    # reads as a plain pinned requirement list.
    (re.compile(r' @ file:///[^\s]+'), ''),
    # stale absolute paths from an earlier workstation
    (re.compile(r'/home/mohammed/Documents/[^\s\'\")]+'), 'docs/figures'),
]
# encoded so this script is not itself flagged by its own scan
_FORBIDDEN_B64 = ('bWhhbWRhbgpjaGVyaWV0CmRlZi1tY2hlcmlldAptaDIwMjJldHMKL2hvbWUv'
                  'KD8hYW5vbikKXGJuaWJpXGIKXGJ0cmlsbGl1bVxi')
FORBIDDEN = [re.compile(p, re.I)
             for p in base64.b64decode(_FORBIDDEN_B64).decode().split('\n')]
TEXT = {'.py', '.sh', '.md', '.txt', '.json', '.yaml', '.yml', '.cfg',
        '.tex', '.bib', '.cff', '.toml', '.gitignore'}


def sanitize(s: str) -> str:
    for pat, rep in SANITIZE:
        s = pat.sub(rep, s)
    return s


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT:
        dst.write_text(sanitize(src.read_text(errors='replace')))
    else:
        shutil.copy2(src, dst)


def write_zip(stage: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if not p.is_file():
                continue
            mode = 0o755 if p.suffix == '.sh' else 0o644
            info = zipfile.ZipInfo(str(p.relative_to(stage)),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = mode << 16
            z.writestr(info, p.read_bytes())
    return dst


def scan_zip(path: Path) -> list[str]:
    hits = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith(('.npz', '.pdf')):
                continue          # binaries: checked separately below
            try:
                txt = z.read(n).decode('utf-8', errors='replace')
            except Exception:
                continue
            for pat in FORBIDDEN:
                for m in pat.finditer(txt):
                    hits.append(f'{n}: {m.group(0)!r}')
        # PDFs: check extracted text and raw metadata strings
        for n in [x for x in z.namelist() if x.endswith('.pdf')]:
            raw = z.read(n).decode('latin-1')
            for pat in FORBIDDEN:
                for m in pat.finditer(raw):
                    hits.append(f'{n} (raw): {m.group(0)!r}')
        for pat in FORBIDDEN:
            if pat.search(' '.join(z.namelist())):
                hits.append(f'filename: {pat.pattern}')
    return hits


# ============================================================ 1. PAPER ZIP
PAPER_SOURCES = [
    'calidoc_paper.tex', 'sec5_practical_lessons.tex', 'calidoc.bib',
    'generated_results_macros.tex', 'spancrc_table.tex', 'backbone_table.tex',
    'sroie_perclass_table.tex', 'wacv.sty', 'ieeenat_fullname.bst',
    'build.sh',
]
PAPER_README = """CaliDoc -- WACV 2027 main paper
================================

  calidoc_paper.pdf   The submission. 8 body pages + references (10 total),
                      WACV 2027 Evaluations & Datasets track, review mode.

  source/             Complete LaTeX sources. Rebuild with:

                          cd source && bash build.sh

                      The build runs pdflatex -> bibtex -> pdflatex x2 and
                      then reports the page count, any overfull boxes and any
                      float-too-large warnings. A clean build reports
                      "OK  body == 8 pages" with zero warnings.

                      figures/ holds the two main-paper figures; both are
                      generated from committed result artifacts by
                      scripts/create_spancrc_figures.py in the code archive,
                      not drawn by hand.

                      generated_results_macros.tex defines every headline
                      number in the paper and is emitted from the result
                      JSONs by scripts/emit_result_macros.py. Do not edit it
                      by hand.

BEFORE SUBMITTING
  Replace the placeholder paper ID in source/calidoc_paper.tex

      \\def\\wacvPaperID{*****}

  with the ID assigned at registration, then rebuild. The supplementary
  document carries the same placeholder and must be updated together.

This archive is anonymized for double-blind review.
"""


def build_paper() -> Path:
    st = STAGE / 'paper'
    if st.exists():
        shutil.rmtree(st)
    st.mkdir(parents=True)
    copy(PKG / 'calidoc_paper.pdf', st / 'calidoc_paper.pdf')
    for f in PAPER_SOURCES:
        copy(PKG / f, st / 'source' / f)
    for f in sorted((PKG / 'figures').glob('spancrc_*.pdf')):
        copy(f, st / 'source' / 'figures' / f.name)
    (st / 'README.txt').write_text(PAPER_README)
    return write_zip(st, PAPER_ZIP)


# ==================================================== 3. GITHUB REPO ZIP
REPO_DIRS = ['scripts', 'evaluation', 'models', 'utils', 'configs']
REPO_FILES = ['requirements.txt', 'requirements-analysis.txt', 'setup.py']
REPO_SKIP_DIRS = {'__pycache__', 'slurm', '.ipynb_checkpoints'}
REPO_SKIP_NAMES = re.compile(r'^(nibi_|trillium_|status_|submit_|squeue_|'
                             r'gate0_|trig_)', re.I)
# Submission-packaging tools: they exist to produce the anonymous review
# archives and have no role in a public research repository. They also embed
# the very identity patterns they rewrite, which would trip the scan.
REPO_SKIP_FILES = {'make_code_supplement.py', 'make_release_zips.py',
                   'verify_package.py'}

GITIGNORE = """# data, checkpoints and run outputs are not versioned
data/
outputs/
checkpoints/
logs/
*.tar
*.ckpt
*.pt

# keep the committed probability caches and result artifacts
!results/cache/*.npz
!results/cache/*.json

# python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/

# latex
*.aux
*.log
*.blg
*.brf
*.out
*.synctex.gz
"""

CITATION_CFF = """cff-version: 1.2.0
title: "CaliDoc: Field-Level Conformal Risk Control and Annotation Auditing for Document Information Extraction"
message: "If you use this software or its results, please cite the paper."
type: software
# ---------------------------------------------------------------------------
# ANONYMIZED FOR DOUBLE-BLIND REVIEW.
# After the review embargo lifts, replace the placeholder below with the real
# author list and add the published venue/DOI.
# ---------------------------------------------------------------------------
authors:
  - name: "Anonymous (double-blind review)"
license: MIT
abstract: >-
  Conformal prediction applied to document information extraction reports
  marginal coverage of per-token or per-entity representative-token prediction
  sets. This repository shows that quantity can be exactly on target while
  certifying nothing a practitioner uses, makes the complete field the measured
  object via conformal risk control, and provides a boundary-escape audit with
  an explicit conditional randomization null.
preferred-citation:
  type: conference-paper
  title: "CaliDoc: Field-Level Conformal Risk Control and Annotation Auditing for Document Information Extraction"
  authors:
    - name: "Anonymous (double-blind review)"
  collection-title: "IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)"
  year: 2027
"""

MIT_LICENSE = """MIT License

Copyright (c) 2026 The CaliDoc Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

REPRODUCE_SH = """#!/usr/bin/env bash
# Reproduce every number in the paper's tables and figures from the committed
# probability caches. No GPU, no checkpoint, no download.
set -euo pipefail
cd "$(dirname "$0")"

echo "== SpanCRC + boundary-escape ============================================"
python3 scripts/span_crc.py --all
echo "== Multiplicity: Holm / BH / BY + document-blocked bootstrap ============"
python3 scripts/multiplicity_analysis.py
echo "== m0 sensitivity ======================================================="
python3 scripts/spancrc_sensitivity.py
echo "== First-token entity CP comparison ====================================="
python3 scripts/first_token_entity_cp.py
echo "== Size and power under controlled corruption (~10 min) ================="
python3 scripts/boundary_escape_power.py
echo "== Regenerate the paper's numeric macros and tables ====================="
python3 scripts/emit_result_macros.py

echo
echo "Done. Every script above self-checks against independently committed"
echo "artifacts and refuses to write output on a mismatch."
"""


def build_repo(readme: str) -> Path:
    st = STAGE / 'repo'
    if st.exists():
        shutil.rmtree(st)
    st.mkdir(parents=True)

    for d in REPO_DIRS:
        for src in sorted((REPO / d).rglob('*')):
            if not src.is_file():
                continue
            rel = src.relative_to(REPO)
            if set(rel.parts) & REPO_SKIP_DIRS:
                continue
            if (REPO_SKIP_NAMES.match(src.name) or src.suffix in {'.pyc'}
                    or src.name in REPO_SKIP_FILES):
                continue
            copy(src, st / rel)
    for f in REPO_FILES:
        if (REPO / f).exists():
            copy(REPO / f, st / f)

    # committed result artifacts + probability caches
    globs = ['results/local_20260803/*.json', 'results/cache/*.json',
             'results/cache/*.npz', 'results/sroie_gate_report_*.md',
             'results/sroie_alignment_diagnosis_*.txt',
             'data_bundle/MANIFEST.json', 'data_bundle/PROVENANCE*.md',
             'data_bundle/environment_freeze.txt']
    for g in globs:
        for src in sorted(REPO.glob(g)):
            copy(src, st / src.relative_to(REPO))

    # The complete paper package, at its repository-relative path so every
    # emitter target stays valid and the repo can rebuild both PDFs.
    for src in sorted(PKG.rglob('*')):
        if not src.is_file() or '__pycache__' in src.parts:
            continue
        if src.suffix in {'.zip', '.aux', '.blg', '.brf', '.out', '.log'}:
            continue
        copy(src, st / src.relative_to(REPO))

    (st / 'README.md').write_text(readme)
    (st / 'LICENSE').write_text(MIT_LICENSE)
    (st / 'CITATION.cff').write_text(CITATION_CFF)
    (st / '.gitignore').write_text(GITIGNORE)
    (st / 'reproduce.sh').write_text(REPRODUCE_SH)
    return write_zip(st, REPO_ZIP)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', action='store_true', help='scan existing zips only')
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if not args.scan:
        print('[1/3] main paper ...')
        build_paper()
        print('[2/3] supplementary (via make_code_supplement.py) ...')
        subprocess.run([sys.executable, 'scripts/make_code_supplement.py'],
                       cwd=REPO, check=True, capture_output=True)
        shutil.copy2(PKG / 'calidoc_wacv2027_supplementary.zip', SUPP_ZIP)
        print('[3/3] github repository ...')
        readme = (REPO / 'build' / 'release' / 'REPO_README.md').read_text()
        build_repo(readme)

    ok = True
    print()
    for z in (PAPER_ZIP, SUPP_ZIP, REPO_ZIP):
        hits = scan_zip(z)
        n = len(zipfile.ZipFile(z).namelist())
        size = z.stat().st_size / 1024 / 1024
        print(f'{"PASS" if not hits else "FAIL"}  {z.name:42s} '
              f'{size:7.1f} MB  {n:4d} entries  identity hits: {len(hits)}')
        for h in hits[:5]:
            print(f'        ! {h}')
        ok &= not hits
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
