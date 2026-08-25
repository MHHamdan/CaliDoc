#!/bin/bash
# Round 2: finish venv deps (pyarrow comes from the arrow MODULE, never pip),
# pre-cache LayoutLMv3 for offline compute nodes, run verify_bundle smoke.
# Executed ON Trillium via `ssh ... 'bash -s'`; output captured on lytos.
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0 2>&1
source /scratch/hamdan/CaliDoc/venv/bin/activate
export PIP_CACHE_DIR=/scratch/hamdan/.pip_cache

echo "=== install remaining deps (NO pyarrow via pip) ==="
pip install --no-index torch torchvision transformers datasets pandas Pillow \
    tqdm accelerate hydra-core omegaconf 2>&1 | tail -4

echo "=== pyarrow must resolve from the arrow module ==="
python -c "import pyarrow; print('pyarrow', pyarrow.__version__, 'from', pyarrow.__file__)"

echo "=== pre-cache LayoutLMv3 to scratch (compute nodes are offline) ==="
export HF_HOME=/scratch/hamdan/CaliDoc/hf_cache
python - <<'EOF'
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
p = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
m = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=54)
print("cached processor + base weights OK")
EOF

echo "=== verify_bundle smoke on Trillium bundle ==="
cd /scratch/hamdan/CaliDoc/code
python - <<'EOF'
import sys, logging
logging.disable(logging.INFO)
sys.path.insert(0, '.')
import torch, transformers
print('torch', torch.__version__, '| cuda build', torch.version.cuda,
      '| transformers', transformers.__version__)
from calidoc.data import verify_bundle, FUNSDDataset
for ds in ('cord', 'funsd'):
    m = verify_bundle('/scratch/hamdan/CaliDoc/data/data_bundle', ds)
    print(ds, m['split_sizes'], '| b_only:', m['b_only_fields'])
# One real __getitem__ through the processor on the ES split
d = FUNSDDataset('/scratch/hamdan/CaliDoc/data/data_bundle', split='es_holdout')
enc = d[0]
print('es_holdout[0] shapes:', {k: tuple(v.shape) for k, v in enc.items()})
print('VERIFY_BUNDLE_OK')
EOF
echo "=== done ==="
