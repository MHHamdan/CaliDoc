#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=calidoc_cord
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/cord_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/cord_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1

# CaliDoc Training Script - CORD Dataset
# Receipt parsing with 21 semantic field labels

echo "=========================================="
echo "CaliDoc CORD Training Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

# Create output directories
mkdir -p /scratch/$USER/calidoc/outputs/logs

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
pip install seqeval

# Set environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Navigate to project directory
cd /scratch/$USER/calidoc

echo "Starting CORD training..."

# Run training
srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=cord \
    training.epochs=20 \
    training.batch_size=8 \
    training.gradient_accumulation_steps=4 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=cord_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/cord_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
