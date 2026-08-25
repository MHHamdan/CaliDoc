#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=funsd_combined
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/funsd_combined_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/funsd_combined_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

# CaliDoc Training - FUNSD Combined Best Strategies
# Target: F1 >= 90% (SOTA: 92.15%)
# Strategy: Combine label smoothing + focal loss + LLRD + longer training

echo "=========================================="
echo "CaliDoc FUNSD Combined Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

# Create directories
mkdir -p /scratch/$USER/calidoc/outputs/logs

# Define resource monitoring log
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/gpu_utilizations_funsd_combined_${SLURM_JOB_ID}.log"
echo "Resource monitoring log: $RESOURCE_LOG"

# Start GPU/CPU monitoring in background (every 5 minutes = 300 seconds)
(
    echo "=== GPU/CPU Resource Monitoring ===" > $RESOURCE_LOG
    echo "Job ID: ${SLURM_JOB_ID}" >> $RESOURCE_LOG
    echo "Node: ${SLURMD_NODENAME}" >> $RESOURCE_LOG
    echo "Script: train_funsd_combined.sh" >> $RESOURCE_LOG
    echo "" >> $RESOURCE_LOG

    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG

        echo "--- GPU Utilization (nvidia-smi) ---" >> $RESOURCE_LOG
        nvidia-smi >> $RESOURCE_LOG 2>&1
        echo "" >> $RESOURCE_LOG

        echo "--- GPU Memory Details ---" >> $RESOURCE_LOG
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu --format=csv >> $RESOURCE_LOG 2>&1
        echo "" >> $RESOURCE_LOG

        echo "--- CPU Usage ---" >> $RESOURCE_LOG
        top -bn1 | head -15 >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG

        echo "--- Memory Usage ---" >> $RESOURCE_LOG
        free -h >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG

        echo "--- Disk Usage ---" >> $RESOURCE_LOG
        df -h /scratch/$USER 2>/dev/null >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG

        sleep 300  # Every 5 minutes
    done
) &
MONITOR_PID=$!
echo "Started GPU monitor with PID: $MONITOR_PID"

# Cleanup function
cleanup() {
    echo "Stopping GPU monitor..."
    kill $MONITOR_PID 2>/dev/null
    echo "" >> $RESOURCE_LOG
    echo "=== Final GPU Stats ===" >> $RESOURCE_LOG
    nvidia-smi >> $RESOURCE_LOG 2>&1
    echo "Job ended at: $(date)" >> $RESOURCE_LOG
}
trap cleanup EXIT

# ============================================
# CRITICAL: Load modules BEFORE creating venv
# Arrow must be loaded before venv for pyarrow
# ============================================
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

echo "Loaded modules:"
module list

# Create virtual environment
echo "Creating virtual environment..."
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Python: $(which python)"
echo "Pip: $(which pip)"

# CRITICAL: Set pip cache to scratch to avoid home quota issues
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

# Upgrade pip
pip install --no-index --upgrade pip

# Install dependencies using cluster wheels
echo "Installing dependencies..."
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

echo ""
echo "Environment setup complete."
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo ""

cd /scratch/$USER/calidoc

echo "Starting FUNSD combined training..."
echo ""

# Combined strategy:
# - LR 5e-5: Proven effective
# - LLRD 0.85: Slightly more aggressive (15% per layer)
# - Label smoothing 0.1: Prevents overconfidence
# - Focal loss gamma 1.0: Mild focus on hard examples
# - 150 epochs: FUNSD is tiny, needs more epochs
srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=funsd \
    training.epochs=150 \
    training.batch_size=4 \
    training.gradient_accumulation_steps=8 \
    training.optimizer.learning_rate=5e-5 \
    training.optimizer.weight_decay=0.01 \
    training.scheduler.warmup_ratio=0.15 \
    +training.optimizer.llrd_rate=0.85 \
    +training.label_smoothing=0.1 \
    +training.use_focal_loss=true \
    +training.focal_gamma=1.0 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=funsd_combined_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/funsd_combined_${SLURM_JOB_ID}

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
