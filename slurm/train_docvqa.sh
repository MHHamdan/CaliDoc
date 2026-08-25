#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=calidoc_docvqa
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/docvqa_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/docvqa_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1

# CaliDoc Training Script - DocVQA Dataset
# Document Visual Question Answering (39K+ training samples)

echo "=========================================="
echo "CaliDoc DocVQA Training Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

# Create output directories
mkdir -p /scratch/$USER/calidoc/outputs/logs

# Resource monitoring
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/docvqa_train_${SLURM_JOB_ID}.log"
(
    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG
        nvidia-smi >> $RESOURCE_LOG 2>&1
        free -h >> $RESOURCE_LOG
        sleep 600
    done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# Load modules BEFORE creating virtualenv
# Arrow module provides pyarrow - must be loaded first!
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

# Create virtual environment in SLURM_TMPDIR (fast local storage)
echo "Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio transformers

# Datasets - use --no-deps to avoid pyarrow conflict
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx

pip install --no-index hydra-core omegaconf accelerate tqdm Pillow scikit-learn
pip install --no-index huggingface_hub requests filelock pyyaml
pip install seqeval

# For DocVQA ANLS metric
pip install python-Levenshtein 2>/dev/null || echo "Levenshtein not available"

# Set environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Navigate to project directory
cd /scratch/$USER/calidoc

# Verify preprocessed data exists
echo "Verifying DocVQA preprocessed data..."
if [ -d "/scratch/$USER/calidoc/data/docvqa/train_merged" ]; then
    echo "Found train_merged:"
    ls -la /scratch/$USER/calidoc/data/docvqa/train_merged/ | head -5
else
    echo "ERROR: train_merged not found at /scratch/$USER/calidoc/data/docvqa/train_merged"
    echo "Run the OCR preprocessing job first!"
    exit 1
fi

if [ -d "/scratch/$USER/calidoc/data/docvqa/val_merged" ]; then
    echo "Found val_merged:"
    ls -la /scratch/$USER/calidoc/data/docvqa/val_merged/ | head -5
else
    echo "ERROR: val_merged not found"
    exit 1
fi

echo "Starting DocVQA training..."
nvidia-smi

# Run training (smaller batch due to larger dataset)
srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=docvqa \
    training.epochs=10 \
    training.batch_size=4 \
    training.gradient_accumulation_steps=8 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=docvqa_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/docvqa_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
