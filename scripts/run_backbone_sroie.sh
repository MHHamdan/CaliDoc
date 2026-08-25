#!/bin/bash
# SROIE on LayoutLMv3-BASE: does the catastrophic token-vs-field proxy
# failure survive the model-scale change? Same protocol as the Large lineage.
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1

TAG=base_local_20260815
NAME="sroie_${TAG}_s42"
LOGDIR=logs/backbone_generality
mkdir -p "$LOGDIR"

echo "=== [$(date -Is)] START sroie (layoutlmv3-base) ===" >> "$LOGDIR/sroie.log"
python3 scripts/train.py \
    training=sroie_phase2 dataset=sroie \
    model.pretrained_name=microsoft/layoutlmv3-base \
    experiment.seed=42 experiment.name="$NAME" >> "$LOGDIR/sroie.log" 2>&1
rc=$?
echo "=== [$(date -Is)] DONE sroie exit=$rc ===" >> "$LOGDIR/sroie.log"
[ $rc -ne 0 ] && { echo "SROIE TRAIN FAILED" >> "$LOGDIR/chain.log"; exit $rc; }

python3 scripts/cache_probs.py \
    --model_path "outputs/$NAME/best" \
    --dataset sroie --bundle_dir ./data_bundle \
    --tag "$TAG" >> "$LOGDIR/sroie.log" 2>&1 || exit 1

python3 scripts/span_crc.py --dataset sroie --cache_tag "$TAG" \
    --results_dir results/backbone_generality --no_self_check \
    >> "$LOGDIR/eval.log" 2>&1

echo "=== [$(date -Is)] SROIE BASE COMPLETE ===" >> "$LOGDIR/chain.log"
