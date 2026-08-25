#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=rebuttal_gpu
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/rebuttal_gpu_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/rebuttal_gpu_%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# CaliDoc CVPR Rebuttal — GPU Experiments
# Runs: RAPS baseline, MC Dropout baseline, Coverage under perturbations
# for both FUNSD and CORD
#
# Usage:
#   sbatch scripts/slurm/run_gpu_rebuttal.sh
#
# Or run individual experiments:
#   sbatch scripts/slurm/run_gpu_rebuttal.sh raps
#   sbatch scripts/slurm/run_gpu_rebuttal.sh mc_dropout
#   sbatch scripts/slurm/run_gpu_rebuttal.sh perturbations

EXPERIMENT=${1:-"all"}

echo "==========================================="
echo "CaliDoc Rebuttal GPU Experiments"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Experiment: $EXPERIMENT"
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

# Set pip cache
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

# Try installing perturbation deps (albumentations/opencv)
pip install --no-index opencv-python-headless albumentations 2>/dev/null || \
    echo "WARNING: albumentations/opencv not available via --no-index, trying pip..."
pip install opencv-python-headless albumentations 2>/dev/null || \
    echo "WARNING: perturbation experiments may fail without albumentations"

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
mkdir -p $MPLCONFIGDIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

# Checkpoint paths
CORD_CKPT="/scratch/$USER/calidoc/outputs/cord_optimized_6530869/best"
FUNSD_CKPT="/scratch/$USER/calidoc/outputs/funsd_aug_6631427/best"
DATA_ROOT="/scratch/$USER/calidoc/data"

# Verify checkpoints exist
echo ""
echo "Checking checkpoints..."
for CKPT in "$CORD_CKPT" "$FUNSD_CKPT"; do
    if [ -f "$CKPT/pytorch_model.bin" ] || [ -f "$CKPT/model.safetensors" ]; then
        echo "  ✓ Found: $CKPT"
    else
        echo "  ✗ MISSING weights in: $CKPT"
        echo "    Contents: $(ls $CKPT/ 2>/dev/null || echo 'directory not found')"
    fi
done
echo ""

# ============================================================
# EXPERIMENT 1: RAPS Baseline
# ============================================================
run_raps() {
    echo "======================================================================"
    echo "RAPS BASELINE EVALUATION"
    echo "======================================================================"

    for DATASET in funsd cord; do
        if [ "$DATASET" = "cord" ]; then
            CKPT="$CORD_CKPT"
        else
            CKPT="$FUNSD_CKPT"
        fi

        echo ""
        echo "--- RAPS: $DATASET ---"
        srun python scripts/evaluate_raps.py \
            --checkpoint_path "$CKPT" \
            --dataset "$DATASET" \
            --data_root "$DATA_ROOT" \
            --output_dir "/scratch/$USER/calidoc/results/raps" \
            --batch_size 8 \
            --num_workers 4 \
            --device cuda \
            --seed 42 \
            --lambda_values 0.001 0.01 0.1 \
            --k_reg_values 1 3 5

        echo ">>> RAPS $DATASET done at $(date)"
    done
}

# ============================================================
# EXPERIMENT 2: MC Dropout
# ============================================================
run_mc_dropout() {
    echo "======================================================================"
    echo "MC DROPOUT EVALUATION"
    echo "======================================================================"

    for DATASET in funsd cord; do
        if [ "$DATASET" = "cord" ]; then
            CKPT="$CORD_CKPT"
        else
            CKPT="$FUNSD_CKPT"
        fi

        echo ""
        echo "--- MC Dropout: $DATASET (T=10) ---"
        srun python scripts/evaluate_mc_dropout.py \
            --checkpoint_path "$CKPT" \
            --dataset "$DATASET" \
            --data_root "$DATA_ROOT" \
            --output_dir "/scratch/$USER/calidoc/results/mc_dropout" \
            --batch_size 2 \
            --num_workers 0 \
            --device cuda \
            --T 10 \
            --seed 42

        echo ">>> MC Dropout $DATASET done at $(date)"
    done
}

# ============================================================
# EXPERIMENT 3: Coverage Under Perturbations
# ============================================================
run_perturbations() {
    echo "======================================================================"
    echo "COVERAGE UNDER PERTURBATIONS"
    echo "======================================================================"

    for DATASET in funsd cord; do
        if [ "$DATASET" = "cord" ]; then
            CKPT="$CORD_CKPT"
        else
            CKPT="$FUNSD_CKPT"
        fi

        echo ""
        echo "--- Perturbations: $DATASET ---"
        srun python scripts/evaluate_coverage_under_perturbations.py \
            --checkpoint_path "$CKPT" \
            --dataset "$DATASET" \
            --data_root "$DATA_ROOT" \
            --output_dir "/scratch/$USER/calidoc/results/coverage_under_perturbations" \
            --batch_size 4 \
            --num_workers 0 \
            --device cuda \
            --alpha 0.10 \
            --seed 42

        echo ">>> Perturbations $DATASET done at $(date)"
    done
}

# ============================================================
# Dispatch
# ============================================================
case "$EXPERIMENT" in
    raps)
        run_raps
        ;;
    mc_dropout)
        run_mc_dropout
        ;;
    perturbations)
        run_perturbations
        ;;
    all)
        run_raps
        run_mc_dropout
        run_perturbations
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: sbatch run_gpu_rebuttal.sh [raps|mc_dropout|perturbations|all]"
        exit 1
        ;;
esac

echo ""
echo "==========================================="
echo "GPU experiments completed at: $(date)"
echo "==========================================="
echo ""
echo "Results in:"
echo "  /scratch/$USER/calidoc/results/raps/"
echo "  /scratch/$USER/calidoc/results/mc_dropout/"
echo "  /scratch/$USER/calidoc/results/coverage_under_perturbations/"
echo ""
echo "To pull results back:"
echo "  scp -r hamdan@nibi.alliancecan.ca:/scratch/hamdan/calidoc/results/{raps,mc_dropout,coverage_under_perturbations} results/"
