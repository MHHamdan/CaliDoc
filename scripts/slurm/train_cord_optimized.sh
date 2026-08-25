#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=cord_opt
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/cord_opt_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/cord_opt_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

# CaliDoc OPTIMIZED Training Script - CORD Dataset
# Target: F1 >= 95% (SOTA: 97.46%)

echo "=========================================="
echo "CaliDoc CORD Optimized Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

# Create virtual environment
echo "Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio transformers
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index hydra-core omegaconf accelerate tqdm Pillow scikit-learn
pip install seqeval  # Critical for F1 computation!

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1

cd /scratch/$USER/calidoc

echo "Starting CORD optimized training..."

# Optimized settings for SOTA F1
# - 30 epochs (CORD has 800 train samples)
# - Learning rate 2e-5 (more stable for 53 labels)
# - Batch size 8 with grad accum 4 = effective 32
srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=cord \
    training.epochs=30 \
    training.batch_size=8 \
    training.gradient_accumulation_steps=4 \
    training.optimizer.learning_rate=2e-5 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=cord_optimized_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/cord_optimized_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
