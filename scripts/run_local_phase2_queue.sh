#!/bin/bash
# Master GPU queue (lytos, GPU0, strictly serial).
#
# Single owner of GPU0 so nothing contends: at the pinned protocol
# LayoutLMv3-large peaks near 44 GB and GPU1 is unavailable (ollama).
# Every step is idempotent -- existing outputs are skipped -- so this can be
# killed and relaunched safely.
#
# Order is deliberate: the multi-seed lineage decides the paper's headline
# numbers, so it runs BEFORE the E1 noise sweep.
#
#   A  seed-42 core: conformal eval -> prob cache -> ablations -> conditional CP
#   B  multi-seed: train seeds 123/456 for all four datasets + conformal eval
#   C  aggregate the 3-seed mean+-std summary
#   D  E7: repaired-SROIE train + full audit
#   E  E1: full fine-tuning under planted label noise
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

TAG=local_20260803
BUNDLE=./data_bundle
REPAIRED=./data_bundle_sroie_repaired
RES=results/$TAG
LOG=logs/$TAG
mkdir -p "$RES" "$LOG" results/cache

say() { echo "=== [$(date -Is)] $*" | tee -a "$LOG/queue.log"; }

# Never start while a training launched outside this queue is still running.
while pgrep -f "python3 scripts/train.py" > /dev/null 2>&1; do sleep 30; done
say "GPU free; master queue starting"

TRAINCFG_cord=cord_phase2
TRAINCFG_funsd=funsd_phase2
TRAINCFG_wildreceipt=wildreceipt_phase2
TRAINCFG_sroie=sroie_phase2

# ------------------------------------------------------------------ phase A
for DS in cord funsd wildreceipt sroie; do
    CK="outputs/${DS}_retrain_s42_local_20260803/best"
    [ -d "$CK" ] || { say "SKIP $DS seed42 (no checkpoint)"; continue; }

    [ -f "$RES/phase2_${DS}.json" ] || {
        say "A: conformal eval $DS s42"
        python3 scripts/evaluate_conformal_phase2.py --model_path "$CK" \
            --dataset "$DS" --bundle_dir "$BUNDLE" \
            --output "$RES/phase2_${DS}.json" --batch_size 8 \
            >> "$LOG/eval_${DS}.log" 2>&1 || say "FAILED eval $DS"; }

    [ -f "results/cache/${DS}_${TAG}.npz" ] || {
        say "A: cache probs $DS"
        python3 scripts/cache_probs.py --model_path "$CK" --dataset "$DS" \
            --bundle_dir "$BUNDLE" --tag "$TAG" \
            >> "$LOG/cache_${DS}.log" 2>&1 || say "FAILED cache $DS"; }

    [ -f "$RES/ablations_${DS}.json" ] || {
        say "A: ablations $DS"
        python3 scripts/ablations_phase2.py --model_path "$CK" --dataset "$DS" \
            --bundle_dir "$BUNDLE" --output "$RES/ablations_${DS}.json" \
            --batch_size 8 >> "$LOG/abl_${DS}.log" 2>&1 \
            || say "FAILED ablations $DS"; }

    if [ ! -f "$RES/conditional_cp_${DS}.json" ] \
       && [ -f "results/cache/${DS}_${TAG}.npz" ]; then
        say "A: conditional CP $DS"
        python3 scripts/conditional_cp_baselines.py \
            --cache "results/cache/${DS}_${TAG}.npz" \
            --output "$RES/conditional_cp_${DS}.json" \
            >> "$LOG/condcp_${DS}.log" 2>&1 || say "FAILED condcp $DS"
    fi
done
say "phase A complete"

# ------------------------------------------------------------------ phase B
# Seeds 123/456 so every headline number can be reported as mean +- std over
# three training seeds. Seed 42 is already done.
for SEED in 123 456; do
    for DS in cord funsd wildreceipt sroie; do
        NAME="${DS}_retrain_s${SEED}_local_20260803"
        eval "TCFG=\$TRAINCFG_${DS}"
        if [ ! -d "outputs/$NAME/best" ]; then
            say "B: train $DS seed $SEED"
            python3 scripts/train.py training="$TCFG" dataset="$DS" \
                experiment.seed="$SEED" experiment.name="$NAME" \
                >> "$LOG/${DS}_s${SEED}.log" 2>&1 || say "FAILED train $DS s$SEED"
        fi
        [ -d "outputs/$NAME/best" ] || continue
        [ -f "$RES/phase2_${DS}_s${SEED}.json" ] || {
            say "B: conformal eval $DS seed $SEED"
            python3 scripts/evaluate_conformal_phase2.py \
                --model_path "outputs/$NAME/best" --dataset "$DS" \
                --bundle_dir "$BUNDLE" \
                --output "$RES/phase2_${DS}_s${SEED}.json" --batch_size 8 \
                >> "$LOG/eval_${DS}_s${SEED}.log" 2>&1 \
                || say "FAILED eval $DS s$SEED"; }
    done
done
say "phase B complete"

# ------------------------------------------------------------------ phase C
say "C: aggregating multi-seed summary"
python3 scripts/aggregate_multiseed.py --results_dir "$RES" \
    --output "$RES/multiseed_summary.json" \
    >> "$LOG/multiseed.log" 2>&1 || say "FAILED multiseed aggregation"
say "phase C complete"

# ------------------------------------------------------------------ phase D
NAME=sroie_repaired_s42_local_20260803
if [ ! -d "outputs/$NAME/best" ]; then
    say "D: train repaired SROIE"
    python3 scripts/train.py training=sroie_phase2 dataset=sroie \
        experiment.seed=42 experiment.name="$NAME" \
        paths.bundle_dir="$REPAIRED" \
        >> "$LOG/sroie_repaired_train.log" 2>&1 || say "FAILED repaired train"
fi
if [ -d "outputs/$NAME/best" ]; then
    CK="outputs/$NAME/best"
    [ -f "$RES/phase2_sroie_repaired.json" ] || {
        say "D: conformal eval repaired SROIE"
        python3 scripts/evaluate_conformal_phase2.py --model_path "$CK" \
            --dataset sroie --bundle_dir "$REPAIRED" \
            --output "$RES/phase2_sroie_repaired.json" --batch_size 8 \
            >> "$LOG/sroie_repaired_eval.log" 2>&1 || say "FAILED repaired eval"; }
    [ -f "results/cache/sroie_repaired_${TAG}.npz" ] || {
        say "D: cache probs repaired SROIE"
        python3 scripts/cache_probs.py --model_path "$CK" --dataset sroie \
            --bundle_dir "$REPAIRED" --tag "repaired_${TAG}" \
            >> "$LOG/sroie_repaired_cache.log" 2>&1 || say "FAILED repaired cache"; }
    if [ ! -f "$RES/conditional_cp_sroie_repaired.json" ] \
       && [ -f "results/cache/sroie_repaired_${TAG}.npz" ]; then
        say "D: conditional CP repaired SROIE"
        python3 scripts/conditional_cp_baselines.py \
            --cache "results/cache/sroie_repaired_${TAG}.npz" \
            --output "$RES/conditional_cp_sroie_repaired.json" \
            >> "$LOG/sroie_repaired_condcp.log" 2>&1 || say "FAILED repaired condcp"
    fi
    [ -f "$RES/ablations_sroie_repaired.json" ] || {
        say "D: ablations repaired SROIE"
        python3 scripts/ablations_phase2.py --model_path "$CK" --dataset sroie \
            --bundle_dir "$REPAIRED" \
            --output "$RES/ablations_sroie_repaired.json" --batch_size 8 \
            >> "$LOG/sroie_repaired_abl.log" 2>&1 || say "FAILED repaired abl"; }
fi
say "phase D complete"

# ------------------------------------------------------------------ phase E
run_cell() {
    local NM=$1 EPS=$2 SEED=$3
    say "E: noise full-finetune $NM eps=$EPS seed=$SEED"
    python3 scripts/noise_full_finetune.py --dataset cord --bundle_dir "$BUNDLE" \
        --noise_model "$NM" --eps "$EPS" --seed "$SEED" \
        --out_dir "$RES/noise_ft" >> "$LOG/noise_ft.log" 2>&1 \
        || say "FAILED E1 cell $NM/$EPS/$SEED"
}
run_cell class_targeted 0.0 0
for EPS in 0.02 0.05 0.10 0.20; do
    for SEED in 0 1; do run_cell class_targeted "$EPS" "$SEED"; done
done
for NM in uniform bio_shift; do
    for EPS in 0.05 0.20; do run_cell "$NM" "$EPS" 0; done
done
say "phase E complete"

say "QUEUE COMPLETE"
