#!/bin/bash
# Local Phase 2 retrain chain (lytos, RTX PRO 6000 Blackwell, GPU0).
#
# Cluster /scratch checkpoints are unreachable, so the whole Phase 2 lineage is
# rebuilt locally. Runs strictly SERIALLY on GPU0: at the pinned protocol
# (batch_size=8, grad_accum=4) LayoutLMv3-large peaks near 44 GB, and GPU1 only
# has ~45 GB free (ollama holds ~43 GB), so co-scheduling OOMs. Batch size is
# NOT reduced -- it is pinned by the Phase 1 paper settings.
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # required by reproducibility.deterministic=true
export CUDA_VISIBLE_DEVICES=0

LOGDIR=logs/local_20260803
mkdir -p "$LOGDIR"

# dataset:training_config
QUEUE=(
    "funsd:funsd_phase2"
    "wildreceipt:wildreceipt_phase2"
    "sroie:sroie_phase2"
)

# Wait for any already-running train.py to finish before taking the GPU.
while pgrep -f "python3 scripts/train.py" > /dev/null 2>&1; do sleep 20; done

for entry in "${QUEUE[@]}"; do
    DS="${entry%%:*}"
    TCFG="${entry##*:}"
    NAME="${DS}_retrain_s42_local_20260803"
    LOG="$LOGDIR/${DS}.log"

    echo "=== [$(date -Is)] START $DS ===" >> "$LOG"
    python3 scripts/train.py \
        training="$TCFG" dataset="$DS" \
        experiment.seed=42 experiment.name="$NAME" >> "$LOG" 2>&1
    echo "=== [$(date -Is)] DONE $DS exit=$? ===" >> "$LOG"
done

echo "=== [$(date -Is)] CHAIN COMPLETE ===" >> "$LOGDIR/chain.log"
