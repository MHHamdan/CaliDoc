#!/bin/bash
# Final follow-on: FUNSD training-strategy ablation under the Phase-2 protocol.
#
# The last item on the original "Pending Re-Run" list. The Phase-1 ablation
# compared strategies on the old (leaky) split; this redoes it under the clean
# early-stopping-holdout protocol so the deltas are attributable to the strategy
# rather than to the split.
#
# funsd_phase2 ships LLRD 0.9 + label smoothing 0.1. Each arm toggles exactly
# one of those, so every delta has a single cause.
#
# Waits on the E2 follow-on's completion marker (which itself waits on the
# master queue), keeping a single owner of GPU0 at all times.
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

TAG=local_20260803
RES=results/$TAG
LOG=logs/$TAG
say() { echo "=== [$(date -Is)] $*" | tee -a "$LOG/funsd_ablation.log"; }

until grep -q "E2 COMPLETE" "$LOG/e2.log" 2>/dev/null; do sleep 60; done
say "E2 finished; starting FUNSD training-strategy ablation"

# name:hydra overrides
ARMS=(
    "no_llrd:training.optimizer.llrd_rate=1.0"
    "no_smoothing:training.label_smoothing=0.0"
    "neither:training.optimizer.llrd_rate=1.0 training.label_smoothing=0.0"
    "augmented:training.robustness.augmentation.enabled=true"
)

for entry in "${ARMS[@]}"; do
    ARM="${entry%%:*}"
    OVR="${entry#*:}"
    NAME="funsd_ablate_${ARM}_local_20260803"

    if [ ! -d "outputs/$NAME/best" ]; then
        say "train $ARM ($OVR)"
        # shellcheck disable=SC2086
        python3 scripts/train.py training=funsd_phase2 dataset=funsd \
            experiment.seed=42 experiment.name="$NAME" $OVR \
            >> "$LOG/funsd_ablate_${ARM}.log" 2>&1 || say "FAILED train $ARM"
    fi
    [ -d "outputs/$NAME/best" ] || continue

    if [ ! -f "$RES/phase2_funsd_ablate_${ARM}.json" ]; then
        say "eval $ARM"
        python3 scripts/evaluate_conformal_phase2.py \
            --model_path "outputs/$NAME/best" --dataset funsd \
            --bundle_dir ./data_bundle \
            --output "$RES/phase2_funsd_ablate_${ARM}.json" --batch_size 8 \
            >> "$LOG/funsd_ablate_${ARM}.log" 2>&1 || say "FAILED eval $ARM"
    fi
done

say "FUNSD STRATEGY ABLATION COMPLETE"
