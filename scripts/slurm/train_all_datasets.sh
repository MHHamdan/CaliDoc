#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=calidoc_all
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-2

# CaliDoc Training Script - Array Job for All Datasets
# Runs FUNSD, CORD, and DocVQA experiments

echo "=========================================="
echo "CaliDoc Array Job Training"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

# Dataset configuration based on array task ID
DATASETS=("funsd" "cord" "docvqa")
EPOCHS=(20 20 10)
BATCH_SIZES=(8 8 4)
GRAD_ACCUM=(4 4 8)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
BATCH=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
ACCUM=${GRAD_ACCUM[$SLURM_ARRAY_TASK_ID]}

echo "Training on dataset: ${DATASET}"
echo "Epochs: ${EPOCH}, Batch size: ${BATCH}, Grad accum: ${ACCUM}"

# Create output directories
mkdir -p /scratch/$USER/calidoc/outputs/logs

# ========================================
# Load modules BEFORE creating virtualenv
# Arrow module provides pyarrow - must be loaded first!
# ========================================
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

# Create virtual environment in SLURM_TMPDIR (fast local storage)
echo "Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# ========================================
# Install dependencies using Compute Canada wheels
# IMPORTANT: Use --no-deps for datasets to avoid pyarrow conflict
# ========================================
pip install --no-index --upgrade pip

# Core packages
pip install --no-index torch torchvision torchaudio transformers

# Datasets - use --no-deps to avoid pyarrow installation attempt
pip install --no-deps datasets

# Dataset dependencies (installed separately since we used --no-deps)
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx

# Other required packages
pip install --no-index hydra-core omegaconf accelerate tqdm Pillow scikit-learn

# NER evaluation
pip install seqeval

# For DocVQA ANLS metric (may not be available, that's ok)
pip install python-Levenshtein 2>/dev/null || echo "Levenshtein not available, will use fallback"

# ========================================
# Set environment variables
# ========================================
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Verify setup
echo ""
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
echo ""

# Navigate to project directory
cd /scratch/$USER/calidoc

echo "Starting ${DATASET} training..."

# Run training
srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=${DATASET} \
    training.epochs=${EPOCH} \
    training.batch_size=${BATCH} \
    training.gradient_accumulation_steps=${ACCUM} \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/${DATASET}_${SLURM_ARRAY_JOB_ID}

EXIT_CODE=$?

echo "=========================================="
echo "${DATASET} job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="
