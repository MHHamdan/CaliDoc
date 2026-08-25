#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=cvpr_eval
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/cvpr_eval_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/cvpr_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# CaliDoc - CVPR Complete Evaluation Suite
# Runs all experiments for both FUNSD and CORD datasets:
# 1. Baseline accuracy evaluation
# 2. Conformal prediction (LAC, APS methods)
# 3. Selective prediction with risk-coverage curves
# 4. Calibration analysis (ECE, MCE, reliability diagrams)
# 5. Robustness evaluation (optional, add --run_robustness)

echo "==========================================="
echo "CaliDoc CVPR Complete Evaluation Suite"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "==========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs
mkdir -p /scratch/$USER/calidoc/results

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

echo "Loaded modules:"
module list

# Create virtual environment
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Python: $(which python)"

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
pip install seqeval

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets

# Fix matplotlib/fontconfig warnings
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
mkdir -p $MPLCONFIGDIR

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

# Results directory
RESULTS_BASE="/scratch/$USER/calidoc/results/cvpr_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_BASE"

echo "==========================================="
echo "Results will be saved to: $RESULTS_BASE"
echo "==========================================="

# ============================================
# FUNSD Evaluation
# ============================================
FUNSD_CHECKPOINT="/scratch/$USER/calidoc/outputs/funsd_aug_6631427/best"
if [ -d "$FUNSD_CHECKPOINT" ]; then
    echo ""
    echo "==========================================="
    echo "Evaluating FUNSD..."
    echo "Checkpoint: $FUNSD_CHECKPOINT"
    echo "==========================================="

    python scripts/evaluate_conformal.py \
        --checkpoint_path "$FUNSD_CHECKPOINT" \
        --dataset funsd \
        --data_root /scratch/$USER/calidoc/data \
        --output_dir "$RESULTS_BASE/funsd" \
        --batch_size 8 \
        --num_workers 4 \
        --device cuda

    echo "FUNSD evaluation completed."
else
    echo "WARNING: FUNSD checkpoint not found at $FUNSD_CHECKPOINT"
    echo "Searching for alternative checkpoints..."
    find /scratch/$USER/calidoc/outputs -name "best" -type d | head -5
fi

# ============================================
# CORD Evaluation
# ============================================
CORD_CHECKPOINT="/scratch/$USER/calidoc/outputs/cord_sota_6632041/best"
if [ -d "$CORD_CHECKPOINT" ]; then
    echo ""
    echo "==========================================="
    echo "Evaluating CORD..."
    echo "Checkpoint: $CORD_CHECKPOINT"
    echo "==========================================="

    python scripts/evaluate_conformal.py \
        --checkpoint_path "$CORD_CHECKPOINT" \
        --dataset cord \
        --data_root /scratch/$USER/calidoc/data \
        --output_dir "$RESULTS_BASE/cord" \
        --batch_size 8 \
        --num_workers 4 \
        --device cuda

    echo "CORD evaluation completed."
else
    echo "WARNING: CORD checkpoint not found at $CORD_CHECKPOINT"
    echo "Searching for alternative checkpoints..."
    find /scratch/$USER/calidoc/outputs -name "*cord*" -type d | head -5
fi

# ============================================
# Generate Combined Summary
# ============================================
echo ""
echo "==========================================="
echo "Generating combined summary..."
echo "==========================================="

python -c "
import json
import os
from pathlib import Path

results_base = '$RESULTS_BASE'
summary = {'funsd': {}, 'cord': {}}

for dataset in ['funsd', 'cord']:
    results_dir = Path(results_base) / dataset
    if results_dir.exists():
        # Find the latest results file
        results_files = list(results_dir.glob('results_*.json'))
        if results_files:
            latest = max(results_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                data = json.load(f)

            summary[dataset] = {
                'baseline_f1': data.get('baseline', {}).get('f1', 0) * 100,
                'ece': data.get('calibration', {}).get('ece', 0) * 100,
                'conformal_90_coverage': data.get('conformal', {}).get('coverage_90', {}).get('lac', {}).get('coverage', 0) * 100,
                'conformal_95_coverage': data.get('conformal', {}).get('coverage_95', {}).get('lac', {}).get('coverage', 0) * 100,
                'conformal_99_coverage': data.get('conformal', {}).get('coverage_99', {}).get('lac', {}).get('coverage', 0) * 100,
            }

print('='*60)
print('CVPR EXPERIMENT SUMMARY')
print('='*60)

for dataset, metrics in summary.items():
    if metrics:
        print(f'\n{dataset.upper()}:')
        print(f'  Baseline F1: {metrics[\"baseline_f1\"]:.2f}%')
        print(f'  ECE (calibration): {metrics[\"ece\"]:.2f}%')
        print(f'  Conformal 90% target: {metrics[\"conformal_90_coverage\"]:.1f}% actual')
        print(f'  Conformal 95% target: {metrics[\"conformal_95_coverage\"]:.1f}% actual')
        print(f'  Conformal 99% target: {metrics[\"conformal_99_coverage\"]:.1f}% actual')

# Save combined summary
with open(os.path.join(results_base, 'combined_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nCombined summary saved to: {results_base}/combined_summary.json')
"

echo ""
echo "==========================================="
echo "Evaluation completed at: $(date)"
echo "Results saved to: $RESULTS_BASE"
echo "==========================================="

# List all results
echo ""
echo "Results structure:"
find "$RESULTS_BASE" -type f -name "*.json" -o -name "*.tex"
