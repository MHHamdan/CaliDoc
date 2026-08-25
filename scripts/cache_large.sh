#!/bin/bash
# Pre-cache microsoft/layoutlmv3-large (the Phase 1 paper model for both CORD
# and FUNSD, per outputs/{cord_optimized,funsd_aug}_* configs) into HF_HOME.
# Works on either cluster: same paths, same modules. Run via `ssh ... 'bash -s'`.
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0 2>&1
source /scratch/hamdan/CaliDoc/venv/bin/activate
export HF_HOME=/scratch/hamdan/CaliDoc/hf_cache

python - <<'EOF'
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)
LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large", num_labels=54)
print("cached layoutlmv3-large OK")
EOF

echo "--- offline load check (what the compute node will do):"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'EOF'
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)
LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large", num_labels=54)
LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
print("OFFLINE_LOAD_OK (large model + both processors)")
EOF

echo "--- queue state:"
squeue -u hamdan
echo "CACHE_LARGE_DONE"
