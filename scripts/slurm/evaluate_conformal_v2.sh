#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=conformal_v2
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/conformal_v2_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/conformal_v2_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# CaliDoc - Structure-Aware Conformal Prediction Evaluation (v2)
# Runs: softmax + structure-aware ablation, temperature scaling,
#       per-class coverage, per-token predictions

echo "==========================================="
echo "CaliDoc Conformal v2 Evaluation"
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
pip install --no-index huggingface_hub requests filelock pyyaml
pip install seqeval

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fix matplotlib/fontconfig warnings
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
mkdir -p $MPLCONFIGDIR

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

# Configuration
CHECKPOINT_PATH=${1:-"/scratch/$USER/calidoc/outputs/cord_optimized_6530869/best"}
DATASET=${2:-"cord"}
SEED=${3:-42}
OUTPUT_DIR="/scratch/$USER/calidoc/results/conformal_v2_${DATASET}_${SLURM_JOB_ID}"

echo "==========================================="
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Dataset: $DATASET"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo "==========================================="

# Run structure-aware conformal evaluation
echo "Starting structure-aware conformal evaluation..."
srun python scripts/evaluate_conformal_v2.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset "$DATASET" \
    --data_root /scratch/$USER/calidoc/data \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --num_workers 4 \
    --device cuda \
    --seed "$SEED" \
    --save_per_token \
    --use_structure_aware

echo "==========================================="
echo "Evaluation completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "==========================================="

ls -la "$OUTPUT_DIR"
