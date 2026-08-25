#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=funsd_smooth
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/funsd_smooth_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/funsd_smooth_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

# CaliDoc Training - FUNSD with Label Smoothing
# Target: F1 >= 90% (SOTA: 92.15%)
# Strategy: Label smoothing (0.1) for regularization
# Previous best: 83.24% F1

echo "=========================================="
echo "CaliDoc FUNSD Training with Label Smoothing"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs

# Resource monitoring
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/gpu_utilizations_funsd_smooth_${SLURM_JOB_ID}.log"
echo "Resource monitoring log: $RESOURCE_LOG"

(
    echo "=== GPU/CPU Resource Monitoring ===" > $RESOURCE_LOG
    echo "Job ID: ${SLURM_JOB_ID}" >> $RESOURCE_LOG
    echo "Script: train_funsd_smoothing.sh" >> $RESOURCE_LOG
    echo "" >> $RESOURCE_LOG

    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG
        echo "--- GPU Utilization ---" >> $RESOURCE_LOG
        nvidia-smi >> $RESOURCE_LOG 2>&1
        echo "" >> $RESOURCE_LOG
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
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

echo "Loaded modules:"
module list

# Create virtual environment
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Python: $(which python)"

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

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

echo "Starting FUNSD training with label smoothing..."

# Label smoothing prevents overconfidence and improves generalization
srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=funsd \
    training.epochs=80 \
    training.batch_size=4 \
    training.gradient_accumulation_steps=8 \
    training.optimizer.learning_rate=5e-5 \
    training.scheduler.warmup_ratio=0.1 \
    +training.optimizer.llrd_rate=0.9 \
    +training.label_smoothing=0.1 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=funsd_smooth_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/funsd_smooth_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
