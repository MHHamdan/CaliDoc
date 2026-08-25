#!/bin/bash
# Nibi environment build, executed ON Nibi via `ssh ... 'bash -s'`.
# All Trillium lessons applied in one pass: pyarrow from the arrow module
# (never pip), no HF `datasets` (depends on pyarrow; Phase 2 reads parquet
# via pandas), LayoutLMv3 pre-cached for offline compute nodes.
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0 2>&1

echo "=== venv build at /scratch/hamdan/CaliDoc/venv ==="
if [ -d /scratch/hamdan/CaliDoc/venv ]; then
  echo "venv already exists, reusing"
else
  virtualenv --no-download /scratch/hamdan/CaliDoc/venv
fi
source /scratch/hamdan/CaliDoc/venv/bin/activate
export PIP_CACHE_DIR=/scratch/hamdan/.pip_cache
pip install --no-index --upgrade pip 2>&1 | tail -1
pip install --no-index torch torchvision transformers pandas Pillow \
    tqdm accelerate hydra-core omegaconf 2>&1 | tail -3
pip install --no-index seqeval 2>&1 | tail -1 || pip install seqeval 2>&1 | tail -1

echo "=== import sanity ==="
python -c "
import torch, transformers, pandas, pyarrow, seqeval, hydra
print('torch', torch.__version__, '| cuda build', torch.version.cuda)
print('transformers', transformers.__version__, '| pyarrow', pyarrow.__version__, '(module-provided)')"

echo "=== pre-cache LayoutLMv3 to scratch (compute nodes are offline) ==="
export HF_HOME=/scratch/hamdan/CaliDoc/hf_cache
python - <<'EOF'
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=54)
print("cached processor + base weights OK")
EOF

echo "=== verify_bundle smoke on Nibi bundle ==="
mkdir -p /scratch/hamdan/CaliDoc/logs /scratch/hamdan/CaliDoc/outputs /scratch/hamdan/CaliDoc/results
cd /scratch/hamdan/CaliDoc/code
python - <<'EOF'
import sys, logging
logging.disable(logging.INFO)
sys.path.insert(0, '.')
from calidoc.data import verify_bundle, CORDDataset
for ds in ('cord', 'funsd'):
    m = verify_bundle('/scratch/hamdan/CaliDoc/data/data_bundle', ds)
    print(ds, m['split_sizes'], '| b_only:', m['b_only_fields'])
d = CORDDataset('/scratch/hamdan/CaliDoc/data/data_bundle', split='val_es')
enc = d[0]
print('cord val_es[0] shapes:', {k: tuple(v.shape) for k, v in enc.items()})
print('VERIFY_BUNDLE_OK')
EOF
echo "=== done ==="
