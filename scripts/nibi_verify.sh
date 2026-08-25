#!/bin/bash
# Nibi Gate 0 verification, executed ON Nibi via `ssh ... 'bash -s'`.
# Idempotent: verifies hashes, extracts only if not yet extracted.
cd /scratch/hamdan/CaliDoc 2>/dev/null || { echo "FATAL: /scratch/hamdan/CaliDoc missing"; exit 1; }

echo "=== files present ==="
ls -lh calidoc_code_20260704.tar data/calidoc_data_20260703.tar 2>&1

echo "=== hash check (both must say OK) ==="
sha256sum -c - <<'SUMS' || { echo "HASH_MISMATCH — do not extract"; exit 1; }
992740b3d834d06550433024fc6257d70caec43dc25e7fda775b4794be736185  calidoc_code_20260704.tar
fcedc9e58b6b7ab652beaeadf3463031e504e831eb8aea51c262c24d54d75cc1  data/calidoc_data_20260703.tar
SUMS
echo "HASHES_OK"

echo "=== extract (if needed) ==="
[ -d code ] && echo "code/ already extracted" || tar -xf calidoc_code_20260704.tar
[ -d data/data_bundle ] && echo "data_bundle/ already extracted" || tar -xf data/calidoc_data_20260703.tar -C data

echo "=== spot checks ==="
grep -c "label_list.json" code/calidoc/data/cord_dataset.py
grep -rn "CORD_LABELS" code/calidoc/ || echo "(CORD_LABELS clean)"
ls data/data_bundle/splits/

echo "=== modules (for pinning nibi_train.sbatch) ==="
module avail 2>&1 | grep -iE "StdEnv|gcc/|cuda/|python/3|arrow/" | head -40
echo "--- hidden cuda probe:"
module --force purge; module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0 2>&1 \
  && echo "PINNED_SET_LOADS_ON_NIBI" || echo "pinned set failed - will pin from avail list"

echo "=== queue ==="
squeue -u hamdan
echo "=== NIBI_VERIFY_DONE ==="
