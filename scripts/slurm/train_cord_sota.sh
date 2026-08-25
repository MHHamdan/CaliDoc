#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=cord_sota
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/cord_sota_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/cord_sota_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

# CaliDoc SOTA Training Script - CORD Dataset
# Target: F1 >= 96% (SOTA: 97.46%)
# Enhancements: EMA, LLRD, Label Smoothing, Longer Training

echo "=========================================="
echo "CaliDoc CORD SOTA Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs

# Resource monitoring
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/cord_sota_${SLURM_JOB_ID}.log"
(
    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG
        nvidia-smi >> $RESOURCE_LOG 2>&1
        free -h >> $RESOURCE_LOG
        sleep 300
    done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

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
pip install --no-index huggingface_hub requests filelock pyyaml
pip install seqeval  # Critical for proper F1 computation!

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1

cd /scratch/$USER/calidoc

echo "Starting CORD SOTA training..."
nvidia-smi

# SOTA settings for maximum F1
# Key optimizations:
# 1. EMA (Exponential Moving Average) with decay 0.999 - stabilizes training
# 2. LLRD (Layer-wise Learning Rate Decay) 0.95 - better fine-tuning
# 3. Label smoothing 0.1 - reduces overconfidence
# 4. 50 epochs - CORD is small (800 samples), needs more epochs
# 5. Lower LR 1.5e-5 with warmup 0.15 - more stable convergence
# 6. Effective batch size 32 (8 * 4 gradient accumulation)

srun python scripts/train.py \
    model=layoutlmv3 \
    model.pretrained_name=microsoft/layoutlmv3-base \
    dataset=cord \
    training.epochs=50 \
    training.batch_size=8 \
    training.gradient_accumulation_steps=4 \
    training.optimizer.learning_rate=1.5e-5 \
    training.optimizer.weight_decay=0.01 \
    training.warmup_ratio=0.15 \
    training.max_grad_norm=1.0 \
    training.use_ema=true \
    training.ema_decay=0.999 \
    training.llrd_rate=0.95 \
    training.label_smoothing=0.1 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=cord_sota_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/cord_sota_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
