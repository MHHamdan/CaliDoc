#!/usr/bin/env bash
# CaliDoc Phase 1: verified dataset preparation (runs entirely on lytos).
# Downloads FUNSD + CORD v2 from authoritative sources, verifies integrity,
# converts to the normalized parquet format, generates protocol splits,
# stats, checksums, provenance, and the transfer-ready bundle.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3

echo "=== Step 1/7: download FUNSD (official zip) ==="
bash "$HERE/download_funsd.sh"

echo "=== Step 2/7: download CORD v2 (pinned HF snapshot) ==="
$PY "$HERE/download_cord.py"

echo "=== Step 3/7: preprocess FUNSD ==="
$PY "$HERE/preprocess_funsd.py"

echo "=== Step 4/7: preprocess CORD ==="
$PY "$HERE/preprocess_cord.py"

echo "=== Step 5/7: protocol splits ==="
$PY "$HERE/make_splits.py"

echo "=== Step 6/7: label statistics ==="
$PY "$HERE/compute_stats.py"

echo "=== Step 7/7: round-trip check + manifest ==="
$PY "$HERE/roundtrip_check.py"
$PY "$HERE/build_manifest.py"

echo "=== DONE: bundle at data_bundle/ ==="
