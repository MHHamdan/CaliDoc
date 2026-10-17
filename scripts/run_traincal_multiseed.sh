#!/bin/bash
# Clean-calibration WildReceipt protocol, LayoutLMv3-LARGE, seeds 42/123/456,
# at the PINNED batch 8 x accum 4 (GPU1 fully free) -- uniform protocol, no
# batch deviation. Replaces the interim seed-42 batch-2x16 run.
set -u
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1

LOGDIR=logs/traincal
mkdir -p "$LOGDIR"

for SEED in 42 123 456; do
  TAG="traincal_b8_s${SEED}_local_20260815"
  NAME="wildreceipt_traincal_${TAG}"
  LOG="$LOGDIR/b8_s${SEED}.log"
  echo "=== [$(date -Is)] START traincal s$SEED (large, batch 8x4) ===" >> "$LOG"
  python3 scripts/train.py \
      training=wildreceipt_phase2 dataset=wildreceipt_traincal \
      experiment.seed=$SEED experiment.name="$NAME" >> "$LOG" 2>&1
  rc=$?
  echo "=== [$(date -Is)] DONE s$SEED exit=$rc ===" >> "$LOG"
  [ $rc -ne 0 ] && { echo "FAILED s$SEED" >> "$LOGDIR/multiseed.log"; exit $rc; }
  python3 scripts/cache_probs.py --model_path "outputs/$NAME/best" \
      --dataset wildreceipt_traincal --bundle_dir ./data_bundle \
      --tag "$TAG" >> "$LOG" 2>&1 || exit 1
  rm -rf "outputs/$NAME"/epoch_*
  mkdir -p "results/traincal/b8_s${SEED}"
  python3 scripts/span_crc.py --dataset wildreceipt_traincal --cache_tag "$TAG" \
      --results_dir "results/traincal/b8_s${SEED}" --no_self_check \
      >> "$LOG" 2>&1 || exit 1
done
echo "=== [$(date -Is)] TRAINCAL MULTISEED COMPLETE ===" >> "$LOGDIR/multiseed.log"
