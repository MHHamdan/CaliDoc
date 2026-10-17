#!/bin/bash
# Clean-calibration WildReceipt protocol (reviewer request, 2026-08-15).
#
# LayoutLMv3-LARGE, seed 42, wildreceipt_phase2 recipe. Protocol deviation
# recorded here: per-device batch 2 x grad-accum 16 (effective batch 32
# unchanged; the pinned 8 x 4 peaks near 44 GB and GPU0 currently has only
# ~17 GB free next to an unrelated job). Optimization trajectory is identical
# up to dropout RNG microbatching.
#
# Chain: train (cal docs excluded by the fit-stage guard) -> cache dump
# (cord-style cal+test units) -> SpanCRC + boundary escape on the clean split.
set -u
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TAG=traincal_local_20260815
NAME="wildreceipt_traincal_${TAG}_s42"
LOGDIR=logs/traincal
mkdir -p "$LOGDIR" results/traincal

echo "=== [$(date -Is)] START wildreceipt_traincal (large, batch 2x16, gpu1) ===" >> "$LOGDIR/train.log"
python3 scripts/train.py \
    training=wildreceipt_phase2 dataset=wildreceipt_traincal \
    training.batch_size=2 training.gradient_accumulation_steps=16 \
    experiment.seed=42 experiment.name="$NAME" >> "$LOGDIR/train.log" 2>&1
rc=$?
echo "=== [$(date -Is)] DONE exit=$rc ===" >> "$LOGDIR/train.log"
[ $rc -ne 0 ] && { echo "TRAIN FAILED" >> "$LOGDIR/chain.log"; exit $rc; }

python3 scripts/cache_probs.py \
    --model_path "outputs/$NAME/best" \
    --dataset wildreceipt_traincal --bundle_dir ./data_bundle \
    --tag "$TAG" >> "$LOGDIR/cache.log" 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "CACHE FAILED" >> "$LOGDIR/chain.log"; exit $rc; }

python3 scripts/span_crc.py --dataset wildreceipt_traincal --cache_tag "$TAG" \
    --results_dir results/traincal --no_self_check >> "$LOGDIR/eval.log" 2>&1

echo "=== [$(date -Is)] TRAINCAL CHAIN COMPLETE ===" >> "$LOGDIR/chain.log"
