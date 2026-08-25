#!/bin/bash
# E2 follow-on: label-error detector benchmark, run after the master queue.
#
# Waits on the literal "QUEUE COMPLETE" marker in queue.log rather than on
# pgrep. Two pgrep-based waiters race: each sees an idle GPU in the gap between
# two queue steps and starts contending. The marker only ever appears once.
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

TAG=local_20260803
RES=results/$TAG
LOG=logs/$TAG
say() { echo "=== [$(date -Is)] $*" | tee -a "$LOG/e2.log"; }

until grep -q "QUEUE COMPLETE" "$LOG/queue.log" 2>/dev/null; do sleep 60; done
say "master queue finished; starting E2"

# eps chosen around the real defect rates we measured: CORD's schema bug
# corrupted ~5% of tokens, SROIE's misalignment ~5-6%.
for EPS in 0.02 0.05 0.10 0.20; do
    OUT="$RES/label_error_baselines_cord_eps${EPS}.json"
    [ -f "$OUT" ] && { say "skip eps=$EPS (exists)"; continue; }
    say "E2 cord class_targeted eps=$EPS"
    python3 scripts/label_error_baselines.py \
        --dataset cord --bundle_dir ./data_bundle \
        --noise_model class_targeted --eps "$EPS" --seed 0 \
        --finetune_layers 4 --batch_size 4 \
        --output "$OUT" >> "$LOG/e2_cord.log" 2>&1 \
        || say "FAILED E2 eps=$EPS"
done

# A uniform-noise arm: token-level detectors should do RELATIVELY better here
# than under class-targeted noise, which is the granularity point we are making.
OUT="$RES/label_error_baselines_cord_uniform.json"
[ -f "$OUT" ] || {
    say "E2 cord uniform eps=0.05"
    python3 scripts/label_error_baselines.py \
        --dataset cord --bundle_dir ./data_bundle \
        --noise_model uniform --eps 0.05 --seed 0 \
        --finetune_layers 4 --batch_size 4 \
        --output "$OUT" >> "$LOG/e2_cord.log" 2>&1 || say "FAILED E2 uniform"; }

say "E2 COMPLETE"
