#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=conformal_multiseed
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/conformal_multiseed_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/conformal_multiseed_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# CaliDoc - Multi-Seed Conformal Evaluation
# Runs evaluate_conformal_v2.py with seeds 42, 123, 456
# on both CORD and FUNSD, then aggregates results.

echo "==========================================="
echo "CaliDoc Multi-Seed Conformal Evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "==========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs
mkdir -p /scratch/$USER/calidoc/results

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

# Create virtual environment
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Set pip cache to scratch
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio transformers
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index hydra-core omegaconf accelerate tqdm Pillow scikit-learn
pip install --no-index huggingface_hub requests filelock pyyaml
pip install seqeval

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
mkdir -p $MPLCONFIGDIR

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

# Configuration
CORD_CKPT="/scratch/$USER/calidoc/outputs/cord_optimized_6530869/best"
FUNSD_CKPT="/scratch/$USER/calidoc/outputs/funsd_aug_6631427/best"
DATA_ROOT="/scratch/$USER/calidoc/data"
SEEDS=(42 123 456)
BASE_OUT="/scratch/$USER/calidoc/results/conformal_v2_multiseed"

echo "CORD checkpoint:  $CORD_CKPT"
echo "FUNSD checkpoint: $FUNSD_CKPT"
echo "Seeds: ${SEEDS[*]}"

for DATASET in cord funsd; do
    if [ "$DATASET" = "cord" ]; then
        CKPT="$CORD_CKPT"
    else
        CKPT="$FUNSD_CKPT"
    fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo "Dataset: $DATASET"
    echo "----------------------------------------------------------------------"

    for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${BASE_OUT}/${DATASET}/seed_${SEED}"
        echo ""
        echo ">>> Running seed=$SEED  output=$OUT_DIR"

        srun python scripts/evaluate_conformal_v2.py \
            --checkpoint_path "$CKPT" \
            --dataset "$DATASET" \
            --data_root "$DATA_ROOT" \
            --output_dir "$OUT_DIR" \
            --seed "$SEED" \
            --use_structure_aware \
            --save_per_token \
            --batch_size 8 \
            --num_workers 4 \
            --device cuda

        echo "<<< Finished seed=$SEED"
    done
done

# Aggregate results
echo ""
echo "======================================================================"
echo "Aggregating multi-seed results..."
echo "======================================================================"

python - "$BASE_OUT" <<'PYEOF'
import json, glob, os, sys
import numpy as np
from collections import defaultdict

base = sys.argv[1]

for dataset in ["cord", "funsd"]:
    pattern = os.path.join(base, dataset, "seed_*", "results_v2_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No results found for {dataset}")
        continue

    print(f"\n--- {dataset.upper()} ({len(files)} seeds) ---")

    metrics = defaultdict(list)
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        metrics['f1'].append(r['baseline']['f1'])
        metrics['ece'].append(r['calibration']['ece'])

        for method in ['lac', 'aps']:
            sm = r.get('conformal_softmax', {}).get(method, {})
            for cov_key, vals in sm.items():
                metrics[f'sm_{method}_{cov_key}_coverage'].append(vals['coverage'])
                metrics[f'sm_{method}_{cov_key}_set_size'].append(vals['avg_set_size'])

        sa = r.get('conformal_structure_aware', {})
        for score_type, cov_dict in sa.items():
            for cov_key, vals in cov_dict.items():
                metrics[f'sa_{score_type}_{cov_key}_coverage'].append(vals['coverage'])
                metrics[f'sa_{score_type}_{cov_key}_set_size'].append(vals['avg_set_size'])

    for key, vals in sorted(metrics.items()):
        arr = np.array(vals)
        print(f"  {key}: {arr.mean():.4f} +/- {arr.std():.4f}  (n={len(arr)})")

    agg = {}
    for key, vals in metrics.items():
        arr = np.array(vals)
        agg[key] = {'mean': float(arr.mean()), 'std': float(arr.std()),
                     'values': [float(v) for v in arr]}
    agg_path = os.path.join(base, dataset, "aggregated.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"  Saved: {agg_path}")

print("\nDone.")
PYEOF

echo ""
echo "==========================================="
echo "Multi-seed evaluation completed at: $(date)"
echo "==========================================="
