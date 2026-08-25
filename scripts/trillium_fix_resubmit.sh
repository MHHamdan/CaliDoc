#!/bin/bash
# Fix the layoutlmv3-large cache miss and resubmit funsd_retrain_s42 on
# Trillium. Cache/check run on the CPU login node (shared /scratch); the
# sbatch itself hops to trig-login01 (GPU jobs cannot be submitted elsewhere).
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0
source /scratch/hamdan/CaliDoc/venv/bin/activate
export HF_HOME=/scratch/hamdan/CaliDoc/hf_cache

echo "=== cache layoutlmv3-large ==="
python - <<'EOF'
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)
LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large", num_labels=54)
print("cached layoutlmv3-large OK")
EOF

echo "=== offline load check (what the compute node does) ==="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'EOF'
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)
LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large", num_labels=54)
LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
print("OFFLINE_LOAD_OK")
EOF

echo "=== resubmit funsd_retrain_s42 from trig-login01 ==="
if squeue -u hamdan -h -o %j | grep -qx funsd_retrain_s42; then
  echo "ABORT: funsd_retrain_s42 still in queue"
  exit 1
fi
ssh -o StrictHostKeyChecking=accept-new trig-login01 \
  'cd /scratch/hamdan/CaliDoc/code && sbatch slurm/phase2_jobs/funsd_retrain_s42.sbatch'
echo "TRILLIUM_RESUBMIT_DONE"
