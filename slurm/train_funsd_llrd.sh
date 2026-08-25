#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=funsd_llrd
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/funsd_llrd_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/funsd_llrd_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

# CaliDoc Training - FUNSD with Layer-Wise Learning Rate Decay
# Target: F1 >= 90% (SOTA: 92.15%)

echo "=========================================="
echo "CaliDoc FUNSD Training with LLRD"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

# Create virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1

cd /scratch/$USER/calidoc

echo "Starting FUNSD training with LLRD..."

# LLRD-enhanced settings
# - llrd_rate=0.95 means each layer gets 5% less LR than the one above
# - Lower LR (1e-5) for more stable training with LLRD
# - 60 epochs for thorough convergence
srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=funsd \
    training.epochs=60 \
    training.batch_size=8 \
    training.gradient_accumulation_steps=4 \
    training.optimizer.learning_rate=1e-5 \
    +training.optimizer.llrd_rate=0.95 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=funsd_llrd_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/funsd_llrd_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
