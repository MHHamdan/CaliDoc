#!/bin/bash
# LayoutLMv3-BASE seeds 123/456 on SROIE + FUNSD + WildReceipt (GPU0).
# Protocol identical to the Large lineage and the Base seed-42 runs.
# Per-epoch checkpoints pruned after each run (best/ kept).
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

LOGDIR=logs/backbone_generality
mkdir -p "$LOGDIR"

for SEED in 123 456; do
  for entry in "sroie:sroie_phase2" "funsd:funsd_phase2" "wildreceipt:wildreceipt_phase2"; do
    DS="${entry%%:*}"; TCFG="${entry##*:}"
    TAG="base_s${SEED}_local_20260815"
    NAME="${DS}_${TAG}"
    LOG="$LOGDIR/${DS}_s${SEED}.log"
    echo "=== [$(date -Is)] START $DS s$SEED (base) ===" >> "$LOG"
    python3 scripts/train.py \
        training="$TCFG" dataset="$DS" \
        model.pretrained_name=microsoft/layoutlmv3-base \
        experiment.seed=$SEED experiment.name="$NAME" >> "$LOG" 2>&1
    rc=$?
    echo "=== [$(date -Is)] DONE $DS s$SEED exit=$rc ===" >> "$LOG"
    [ $rc -ne 0 ] && { echo "FAILED $DS s$SEED" >> "$LOGDIR/multiseed.log"; exit $rc; }
    python3 scripts/cache_probs.py --model_path "outputs/$NAME/best" \
        --dataset "$DS" --bundle_dir ./data_bundle --tag "$TAG" >> "$LOG" 2>&1 || exit 1
    rm -rf "outputs/$NAME"/epoch_*
    mkdir -p "results/backbone_generality/s${SEED}"
    python3 scripts/span_crc.py --dataset "$DS" --cache_tag "$TAG" \
        --results_dir "results/backbone_generality/s${SEED}" --no_self_check \
        >> "$LOG" 2>&1 || exit 1
  done
done
echo "=== [$(date -Is)] BASE MULTISEED COMPLETE ===" >> "$LOGDIR/multiseed.log"
