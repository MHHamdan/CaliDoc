#!/bin/bash
# E1 extension: high-epsilon cells.
#
# The master queue's phase E sweeps eps in {0.02..0.20}. That is well BELOW the
# magnitude of the defect this experiment exists to model: the real CORD schema
# bug folded an entire field family into the wrong label, i.e. eps = 1.0 on the
# affected classes, and the SROIE misalignment corrupts ~5-6% of entity tokens.
# A dose-response curve that stops at 0.20 therefore never reaches the observed
# defect, and cannot establish where the diagnostic saturates.
#
# These cells extend the class-targeted arm to eps in {0.40, 0.80, 1.00}, with
# 1.00 reproducing the CORD bug exactly (every token of one field relabelled).
#
# Chained behind the FUNSD strategy ablation so GPU0 keeps a single owner.
set -u
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

TAG=local_20260803
RES=results/$TAG
LOG=logs/$TAG
say() { echo "=== [$(date -Is)] $*" | tee -a "$LOG/e1_high_eps.log"; }

until grep -q "FUNSD STRATEGY ABLATION COMPLETE" "$LOG/funsd_ablation.log" 2>/dev/null; do
    sleep 60
done
say "starting high-epsilon E1 cells"

for EPS in 0.40 0.80 1.00; do
    for SEED in 0 1; do
        say "E1 class_targeted eps=$EPS seed=$SEED"
        python3 scripts/noise_full_finetune.py \
            --dataset cord --bundle_dir ./data_bundle \
            --noise_model class_targeted --eps "$EPS" --seed "$SEED" \
            --out_dir "$RES/noise_ft" >> "$LOG/noise_ft.log" 2>&1 \
            || say "FAILED cell $EPS/$SEED"
    done
done

# bio_shift at high rate too: the SROIE analogue, where boundary corruption is
# the mechanism rather than cross-field confusion.
for EPS in 0.40 0.80; do
    say "E1 bio_shift eps=$EPS seed=0"
    python3 scripts/noise_full_finetune.py \
        --dataset cord --bundle_dir ./data_bundle \
        --noise_model bio_shift --eps "$EPS" --seed 0 \
        --out_dir "$RES/noise_ft" >> "$LOG/noise_ft.log" 2>&1 \
        || say "FAILED bio_shift $EPS"
done

say "E1 HIGH EPS COMPLETE"
