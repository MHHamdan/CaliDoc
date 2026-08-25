#!/usr/bin/env bash
# ============================================================================
# CaliDoc Multi-Seed Conformal Evaluation
#
# Runs evaluate_conformal_v2.py with 3 seeds to measure calibration variance.
# The SAME trained model (seed=42) is used; only the calibration split
# randomness varies — which is the correct approach for conformal prediction
# (the model is fixed, calibration randomness is the source of variance).
#
# Usage:
#   bash scripts/run_conformal_multiseed.sh
#
# Environment:
#   Expects the calidoc conda environment to be activated.
#   Adjust CORD_CKPT / FUNSD_CKPT / DATA_ROOT as needed.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Configuration --------------------------------------------------------
CORD_CKPT="${CORD_CKPT:-outputs/cord_optimized_6530869/best}"
FUNSD_CKPT="${FUNSD_CKPT:-outputs/funsd_aug_6631427/best}"
DATA_ROOT="${DATA_ROOT:-data}"
SEEDS=(42 123 456)
BASE_OUT="results/conformal_v2_multiseed"
# --------------------------------------------------------------------------

echo "======================================================================"
echo "CaliDoc Multi-Seed Conformal Evaluation"
echo "======================================================================"
echo "CORD checkpoint:  $CORD_CKPT"
echo "FUNSD checkpoint: $FUNSD_CKPT"
echo "Seeds: ${SEEDS[*]}"
echo ""

for DATASET in cord funsd; do
    if [ "$DATASET" = "cord" ]; then
        CKPT="$CORD_CKPT"
    else
        CKPT="$FUNSD_CKPT"
    fi

    echo "----------------------------------------------------------------------"
    echo "Dataset: $DATASET"
    echo "----------------------------------------------------------------------"

    for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${BASE_OUT}/${DATASET}/seed_${SEED}"
        echo ""
        echo ">>> Running seed=$SEED  output=$OUT_DIR"

        python scripts/evaluate_conformal_v2.py \
            --checkpoint_path "$CKPT" \
            --dataset "$DATASET" \
            --data_root "$DATA_ROOT" \
            --output_dir "$OUT_DIR" \
            --seed "$SEED" \
            --use_structure_aware \
            --save_per_token

        echo "<<< Finished seed=$SEED"
    done
done

# --- Aggregate results ----------------------------------------------------
echo ""
echo "======================================================================"
echo "Aggregating multi-seed results..."
echo "======================================================================"

python - <<'PYEOF'
import json, glob, os, sys
import numpy as np
from collections import defaultdict

base = "results/conformal_v2_multiseed"

for dataset in ["cord", "funsd"]:
    pattern = os.path.join(base, dataset, "seed_*", "results_v2_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No results found for {dataset}")
        continue

    print(f"\n--- {dataset.upper()} ({len(files)} seeds) ---")

    # Collect metrics across seeds
    metrics = defaultdict(list)
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        metrics['f1'].append(r['baseline']['f1'])
        metrics['ece'].append(r['calibration']['ece'])

        # Softmax conformal
        for method in ['lac', 'aps']:
            sm = r.get('conformal_softmax', {}).get(method, {})
            for cov_key, vals in sm.items():
                metrics[f'sm_{method}_{cov_key}_coverage'].append(vals['coverage'])
                metrics[f'sm_{method}_{cov_key}_set_size'].append(vals['avg_set_size'])

        # Structure-aware conformal
        sa = r.get('conformal_structure_aware', {})
        for score_type, cov_dict in sa.items():
            for cov_key, vals in cov_dict.items():
                metrics[f'sa_{score_type}_{cov_key}_coverage'].append(vals['coverage'])
                metrics[f'sa_{score_type}_{cov_key}_set_size'].append(vals['avg_set_size'])

    # Print summary
    for key, vals in sorted(metrics.items()):
        arr = np.array(vals)
        print(f"  {key}: {arr.mean():.4f} +/- {arr.std():.4f}  (n={len(arr)})")

    # Save aggregated JSON
    agg = {}
    for key, vals in metrics.items():
        arr = np.array(vals)
        agg[key] = {'mean': float(arr.mean()), 'std': float(arr.std()),
                     'values': [float(v) for v in arr]}
    agg_path = os.path.join(base, dataset, "aggregated.json")
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"  Saved: {agg_path}")

print("\nDone.")
PYEOF

echo ""
echo "======================================================================"
echo "Multi-seed evaluation complete."
echo "======================================================================"
