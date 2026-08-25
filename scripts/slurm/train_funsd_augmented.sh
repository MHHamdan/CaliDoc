#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=funsd_aug
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/funsd_aug_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/funsd_aug_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

# CaliDoc Training - FUNSD with Data Augmentation
# Target: F1 >= 90% (SOTA: 92.15%)
# Strategy: Data augmentation + consistency regularization

echo "=========================================="
echo "CaliDoc FUNSD Augmented Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

# Create directories
mkdir -p /scratch/$USER/calidoc/outputs/logs

# Define resource monitoring log
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/gpu_utilizations_funsd_aug_${SLURM_JOB_ID}.log"
echo "Resource monitoring log: $RESOURCE_LOG"

# Start GPU/CPU monitoring in background
(
    echo "=== GPU/CPU Resource Monitoring ===" > $RESOURCE_LOG
    echo "Job ID: ${SLURM_JOB_ID}" >> $RESOURCE_LOG
    echo "Node: ${SLURMD_NODENAME}" >> $RESOURCE_LOG
    echo "Script: train_funsd_augmented.sh" >> $RESOURCE_LOG
    echo "" >> $RESOURCE_LOG

    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG
        echo "--- GPU Utilization ---" >> $RESOURCE_LOG
        nvidia-smi >> $RESOURCE_LOG 2>&1
        echo "" >> $RESOURCE_LOG
        echo "--- GPU Memory Details ---" >> $RESOURCE_LOG
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu --format=csv >> $RESOURCE_LOG 2>&1
        echo "" >> $RESOURCE_LOG
        echo "--- CPU/Memory ---" >> $RESOURCE_LOG
        top -bn1 | head -15 >> $RESOURCE_LOG
        free -h >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG
        sleep 300
    done
) &
MONITOR_PID=$!
echo "Started GPU monitor with PID: $MONITOR_PID"

cleanup() {
    kill $MONITOR_PID 2>/dev/null
    echo "=== Final Stats ===" >> $RESOURCE_LOG
    nvidia-smi >> $RESOURCE_LOG 2>&1
    echo "Job ended at: $(date)" >> $RESOURCE_LOG
}
trap cleanup EXIT

# CRITICAL: Load modules BEFORE creating venv
# opencv must be loaded before venv for cv2 to work
# NOTE: Load opencv AFTER cuda to avoid conflicts
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11 opencv/4.10.0

echo "Loaded modules:"
module list

# Verify opencv module is loaded
echo "Verifying opencv module..."
python -c "import cv2; print('OpenCV version:', cv2.__version__)" || echo "WARNING: cv2 import failed"

# Create virtual environment
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Python: $(which python)"

# CRITICAL: Set pip cache to scratch to avoid home quota issues
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio transformers
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index hydra-core omegaconf accelerate tqdm Pillow scikit-learn
# opencv is loaded via module, no pip install needed
pip install seqeval

# Verify cv2 is available inside venv
echo "Verifying cv2 in venv..."
python -c "import cv2; print('OpenCV in venv:', cv2.__version__)" || {
    echo "ERROR: cv2 not available in venv!"
    echo "Module system cv2 may not be accessible from venv"
    exit 1
}

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

echo "Starting FUNSD augmented training..."

# Data augmentation + regularization for small dataset
srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=funsd \
    training=robustness \
    training.epochs=120 \
    training.batch_size=4 \
    training.gradient_accumulation_steps=8 \
    training.optimizer.learning_rate=5e-5 \
    training.scheduler.warmup_ratio=0.1 \
    +training.optimizer.llrd_rate=0.9 \
    +training.label_smoothing=0.1 \
    training.robustness.augmentation.enabled=true \
    training.robustness.augmentation.probability=0.5 \
    training.robustness.consistency.enabled=true \
    training.robustness.consistency.weight=0.05 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=funsd_aug_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/funsd_aug_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
