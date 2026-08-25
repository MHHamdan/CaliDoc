#!/bin/bash
# E7 step 3 -- train and audit the REPAIRED SROIE lineage.
#
# Runs after the main queue so it does not contend for GPU0. The repaired
# bundle lives in its own root (data_bundle_sroie_repaired) with dataset_name
# still 'sroie', so no loader code changes are needed.
#
# The comparison that matters: per-class conformal coverage on the repaired
# labels vs the upstream pre-tagged labels. If the entity classes that
# collapsed to ~0.00 recover to near-nominal while test F1 barely moves, the
# auditor found a real annotation defect and the loop closes.
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

TAG=local_20260803
BUNDLE=data_bundle_sroie_repaired
RES=results/$TAG
LOG=logs/$TAG
NAME=sroie_repaired_s42_local_20260803
mkdir -p "$RES" "$LOG"

say() { echo "=== [$(date -Is)] $*" | tee -a "$LOG/sroie_repaired.log"; }

# Wait for every other GPU job (training chain + main queue) to drain.
while pgrep -f "scripts/train.py|noise_full_finetune.py|evaluate_conformal_phase2.py|cache_probs.py|ablations_phase2.py" > /dev/null 2>&1; do
    sleep 60
done
say "GPU free; starting repaired-SROIE lineage"

if [ ! -d "outputs/$NAME/best" ]; then
    say "training $NAME on the repaired bundle"
    python3 scripts/train.py training=sroie_phase2 dataset=sroie \
        experiment.seed=42 experiment.name="$NAME" \
        paths.bundle_dir="./$BUNDLE" >> "$LOG/sroie_repaired_train.log" 2>&1 \
        || say "FAILED training"
fi

CK="outputs/$NAME/best"
[ -d "$CK" ] || { say "no checkpoint, aborting"; exit 1; }

if [ ! -f "$RES/phase2_sroie_repaired.json" ]; then
    say "conformal eval (repaired)"
    python3 scripts/evaluate_conformal_phase2.py \
        --model_path "$CK" --dataset sroie --bundle_dir "./$BUNDLE" \
        --output "$RES/phase2_sroie_repaired.json" --batch_size 8 \
        >> "$LOG/sroie_repaired_eval.log" 2>&1 || say "FAILED conformal eval"
fi

if [ ! -f "results/cache/sroie_repaired_${TAG}.npz" ]; then
    say "cache probs (repaired)"
    python3 scripts/cache_probs.py \
        --model_path "$CK" --dataset sroie --bundle_dir "./$BUNDLE" \
        --tag "repaired_${TAG}" >> "$LOG/sroie_repaired_cache.log" 2>&1 \
        || say "FAILED cache"
fi

if [ ! -f "$RES/conditional_cp_sroie_repaired.json" ] \
   && [ -f "results/cache/sroie_repaired_${TAG}.npz" ]; then
    say "conditional CP (repaired)"
    python3 scripts/conditional_cp_baselines.py \
        --cache "results/cache/sroie_repaired_${TAG}.npz" \
        --output "$RES/conditional_cp_sroie_repaired.json" \
        >> "$LOG/sroie_repaired_condcp.log" 2>&1 || say "FAILED conditional CP"
fi

if [ ! -f "$RES/ablations_sroie_repaired.json" ]; then
    say "ablations (repaired)"
    python3 scripts/ablations_phase2.py \
        --model_path "$CK" --dataset sroie --bundle_dir "./$BUNDLE" \
        --output "$RES/ablations_sroie_repaired.json" --batch_size 8 \
        >> "$LOG/sroie_repaired_abl.log" 2>&1 || say "FAILED ablations"
fi

say "SROIE REPAIRED COMPLETE"
