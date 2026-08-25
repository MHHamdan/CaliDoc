#!/bin/bash
# Final stage: coverage under document perturbations, all 9 types x 3 severities.
#
# This closes the last item on the original "Pending Re-Run" list. The Phase-1
# evaluation covered only 4 of 9 perturbations, attributed at the time to a
# missing OpenCV module on the cluster. The real cause was a bug in
# calidoc/data/perturbations.py: only 3 methods called the lazy cv2 loader, so
# the rest used a None handle even where OpenCV was installed. Fixed, so all 9
# now run.
set -u
cd /home/mhamdan/Calidoc
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

TAG=local_20260803
RES=results/$TAG
LOG=logs/$TAG
say() { echo "=== [$(date -Is)] $*" | tee -a "$LOG/perturbation.log"; }

until grep -q "E1 HIGH EPS COMPLETE" "$LOG/e1_high_eps.log" 2>/dev/null; do sleep 60; done
say "starting perturbation-coverage evaluation"

for DS in funsd cord wildreceipt sroie; do
    CK="outputs/${DS}_retrain_s42_local_20260803/best"
    OUT="$RES/perturbation_coverage_${DS}.json"
    [ -d "$CK" ] || { say "SKIP $DS (no checkpoint)"; continue; }
    [ -f "$OUT" ] && { say "skip $DS (exists)"; continue; }
    say "perturbation coverage: $DS"
    python3 scripts/evaluate_perturbation_coverage.py \
        --model_path "$CK" --dataset "$DS" --bundle_dir ./data_bundle \
        --output "$OUT" --batch_size 8 \
        >> "$LOG/perturbation_${DS}.log" 2>&1 || say "FAILED $DS"
done

say "PERTURBATION COVERAGE COMPLETE"
