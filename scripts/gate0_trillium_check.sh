#!/bin/bash
# Gate 0 / code-sync verification, executed ON Trillium via `ssh ... 'bash -s'`.
# Output is captured on lytos; do not run this on lytos directly.
echo "=== hostname ==="; hostname
echo "=== 1) label_list.json refs in cord_dataset.py (must be >=1) ==="
cd /scratch/hamdan/CaliDoc/code 2>/dev/null || { echo "FATAL: code dir missing"; exit 1; }
grep -c "label_list.json" calidoc/data/cord_dataset.py
echo "=== 2) CORD_LABELS occurrences in calidoc/ (must be empty) ==="
grep -rn "CORD_LABELS" calidoc/ || echo "(clean)"
echo "=== 3) corrected files present ==="
ls -la calidoc/data/bundle.py calidoc/data/sroie_dataset.py \
      configs/training/cord_phase2.yaml configs/training/funsd_phase2.yaml 2>&1
echo "--- docvqa remnants in calidoc/data (must be 0):"
ls calidoc/data/ | grep -c docvqa
echo "--- stale top-level training/ on remote (must say missing):"
ls -d training 2>&1
echo "=== 4) bundle splits ==="
ls /scratch/hamdan/CaliDoc/data/data_bundle/splits/
echo "=== 5) modules ==="
module avail 2>&1 | grep -iE "StdEnv|gcc/|cuda/|python/3|arrow/" | head -40
echo "=== 6) queue ==="
squeue -u hamdan
echo "=== done ==="
