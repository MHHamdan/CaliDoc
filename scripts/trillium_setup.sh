#!/bin/bash
# Trillium venv build + verify_bundle smoke, executed ON Trillium via
# `ssh ... 'bash -s'`. Output captured on lytos.
echo "=== module spider cuda (versions) ==="
module spider cuda 2>&1 | head -30
echo "=== load pinned module set ==="
module --force purge
if module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0 2>&1; then
  echo "LOAD_OK: cuda/12.2"
else
  echo "cuda/12.2 failed; trying default cuda"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda python/3.11.5 arrow/17.0.0 2>&1 && echo "LOAD_OK: default cuda"
fi
module list 2>&1

echo "=== venv build at /scratch/hamdan/CaliDoc/venv ==="
if [ -d /scratch/hamdan/CaliDoc/venv ]; then
  echo "venv already exists, reusing"
else
  virtualenv --no-download /scratch/hamdan/CaliDoc/venv
fi
source /scratch/hamdan/CaliDoc/venv/bin/activate
export PIP_CACHE_DIR=/scratch/hamdan/.pip_cache
pip install --no-index --upgrade pip 2>&1 | tail -1
echo "--- installing core deps from cluster wheelhouse"
pip install --no-index torch torchvision transformers datasets pandas pyarrow \
    Pillow scikit-learn tqdm accelerate hydra-core omegaconf 2>&1 | tail -4
echo "--- seqeval (wheelhouse first, PyPI fallback)"
pip install --no-index seqeval 2>&1 | tail -2 || pip install seqeval 2>&1 | tail -2

echo "=== verify_bundle smoke on Trillium bundle (CPU, login node) ==="
cd /scratch/hamdan/CaliDoc/code
python - <<'EOF'
import sys, logging
logging.disable(logging.INFO)
sys.path.insert(0, '.')
import torch, transformers
print('torch', torch.__version__, '| cuda build', torch.version.cuda,
      '| transformers', transformers.__version__)
from calidoc.data import verify_bundle
for ds in ('cord', 'funsd'):
    m = verify_bundle('/scratch/hamdan/CaliDoc/data/data_bundle', ds)
    print(ds, m['split_sizes'], '| b_only:', m['b_only_fields'],
          '| O:', {k: round(v, 4) for k, v in m['o_fraction_per_source_split'].items()})
print('VERIFY_BUNDLE_OK')
EOF
echo "=== done ==="
