#!/bin/bash
# Fix the layoutlmv3-large cache miss and resubmit cord_retrain_s42 on Nibi.
# Gated: resubmits ONLY if the offline load check passes.
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0
source /scratch/hamdan/CaliDoc/venv/bin/activate
export HF_HOME=/scratch/hamdan/CaliDoc/hf_cache

echo "=== state of job 17186418 (expected FAILED from the -large cache miss) ==="
sacct -j 17186418 -o "JobID,State,Elapsed,ExitCode" | head -4
tail -3 /scratch/hamdan/CaliDoc/logs/cord_retrain_s42_17186418.err 2>/dev/null || true
# cancel if somehow still alive (it cannot succeed without the cached model)
if squeue -u hamdan -h -o %A | grep -qx 17186418; then
  echo "still queued/running -> scancel"
  scancel 17186418
  sleep 5
fi

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

echo "=== resubmit cord_retrain_s42 ==="
if squeue -u hamdan -h -o %j | grep -qx cord_retrain_s42; then
  echo "ABORT: cord_retrain_s42 still in queue"
  exit 1
fi
cd /scratch/hamdan/CaliDoc/code
sbatch slurm/phase2_jobs/cord_retrain_s42.sbatch
echo "NIBI_RESUBMIT_DONE"
