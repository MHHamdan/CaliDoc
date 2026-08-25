#!/bin/bash
# E2 re-run with the class-level localisation metric.
#
# The first E2 pass measured token-level AUPRC only, on which a class-granular
# score cannot compete by construction (every token of a class gets the same
# value, so within-class ordering is arbitrary). That table shows our method
# losing and provides no evidence for the actual claim, which is about
# identifying WHICH CLASS is broken.
#
# class_level_ranking() aggregates each detector's token scores to a per-class
# mean and reports the rank of the truly-corrupted class. It is applied
# identically to every baseline, so nothing is handicapped. Both metrics are
# reported; if we lose on both, that is the finding.
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
# The weights are already cached. Without this, every model construction (6 per
# cell -- one per CV fold plus the full-data fit) issues HuggingFace HTTP
# requests, and a stalled fetch has no timeout: the first attempt at this stage
# hung for ~8 hours in futex_wait on an in-flight model.safetensors request.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TAG=local_20260803
RES=results/$TAG
LOG=logs/$TAG
say() { echo "=== [$(date -Is)] $*" | tee -a "$LOG/e2_classlevel.log"; }

until grep -q "PERTURBATION COVERAGE COMPLETE" "$LOG/perturbation.log" 2>/dev/null; do
    sleep 60
done
say "re-running E2 with class-level localisation metric"

for SPEC in "class_targeted:0.02" "class_targeted:0.05" "class_targeted:0.10" \
            "class_targeted:0.20" "uniform:0.05" "bio_shift:0.20"; do
    NM="${SPEC%%:*}"; EPS="${SPEC##*:}"
    OUT="$RES/label_error_baselines_cord_${NM}_eps${EPS}.json"
    [ -f "$OUT" ] && { say "skip $NM/$EPS"; continue; }
    say "E2 class-level: $NM eps=$EPS"
    python3 scripts/label_error_baselines.py \
        --dataset cord --bundle_dir ./data_bundle \
        --noise_model "$NM" --eps "$EPS" --seed 0 \
        --finetune_layers 4 --batch_size 4 \
        --output "$OUT" >> "$LOG/e2_classlevel_run.log" 2>&1 \
        || say "FAILED $NM/$EPS"
done

say "E2 CLASS-LEVEL COMPLETE"
