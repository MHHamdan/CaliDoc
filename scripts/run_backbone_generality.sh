#!/bin/bash
# Second-backbone generality test: LayoutLMv3-BASE on FUNSD + WildReceipt.
#
# Purpose (reviewer request): test whether the token-vs-field measurement
# failure and the SpanCRC repair survive a model-scale change. NOT a SOTA run.
# Protocol identical to the Large lineage (same configs, seed 42, same splits,
# same ES discipline); the only change is model.pretrained_name.
#
# Runs serially on GPU1 (GPU0 is fully booked). Base peaks well under the
# ~20 GB currently free. Cache tag: base_local_20260815.
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1

TAG=base_local_20260815
LOGDIR=logs/backbone_generality
mkdir -p "$LOGDIR" results/backbone_generality

QUEUE=(
    "funsd:funsd_phase2"
    "wildreceipt:wildreceipt_phase2"
)

for entry in "${QUEUE[@]}"; do
    DS="${entry%%:*}"
    TCFG="${entry##*:}"
    NAME="${DS}_${TAG}_s42"
    LOG="$LOGDIR/${DS}.log"

    echo "=== [$(date -Is)] START $DS (layoutlmv3-base) ===" >> "$LOG"
    python3 scripts/train.py \
        training="$TCFG" dataset="$DS" \
        model.pretrained_name=microsoft/layoutlmv3-base \
        experiment.seed=42 experiment.name="$NAME" >> "$LOG" 2>&1
    rc=$?
    echo "=== [$(date -Is)] DONE $DS exit=$rc ===" >> "$LOG"
    [ $rc -ne 0 ] && { echo "TRAIN FAILED $DS" >> "$LOGDIR/chain.log"; exit $rc; }

    echo "=== [$(date -Is)] CACHE $DS ===" >> "$LOG"
    python3 scripts/cache_probs.py \
        --model_path "outputs/$NAME/best" \
        --dataset "$DS" --bundle_dir ./data_bundle \
        --tag "$TAG" >> "$LOG" 2>&1
    rc=$?
    echo "=== [$(date -Is)] CACHE DONE $DS exit=$rc ===" >> "$LOG"
    [ $rc -ne 0 ] && { echo "CACHE FAILED $DS" >> "$LOGDIR/chain.log"; exit $rc; }
done

# CPU evaluation: SpanCRC + boundary escape on the base caches, results kept
# in their own directory so committed Large artifacts are untouched.
python3 scripts/span_crc.py --dataset funsd --cache_tag "$TAG" \
    --results_dir results/backbone_generality --no_self_check \
    >> "$LOGDIR/eval.log" 2>&1
python3 scripts/span_crc.py --dataset wildreceipt --cache_tag "$TAG" \
    --results_dir results/backbone_generality --no_self_check \
    >> "$LOGDIR/eval.log" 2>&1

echo "=== [$(date -Is)] BACKBONE CHAIN COMPLETE ===" >> "$LOGDIR/chain.log"
