#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=funsd_base
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/funsd_base_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/funsd_base_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1

# CaliDoc Training - FUNSD with LayoutLMv3-BASE + EMA
# Target: F1 >= 90% (SOTA: 92.15%)
# Strategy: Smaller model may generalize better on tiny dataset (149 samples)
# Base model: 125M params vs Large: 368M params

echo "=========================================="
echo "CaliDoc FUNSD Training - BASE Model + EMA"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs

# Resource monitoring
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/gpu_utilizations_funsd_base_${SLURM_JOB_ID}.log"

(
    echo "=== GPU/CPU Resource Monitoring ===" > $RESOURCE_LOG
    echo "Job ID: ${SLURM_JOB_ID}" >> $RESOURCE_LOG
    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG
        nvidia-smi >> $RESOURCE_LOG 2>&1
        echo "" >> $RESOURCE_LOG
        sleep 300
    done
) &
MONITOR_PID=$!

cleanup() {
    kill $MONITOR_PID 2>/dev/null
    echo "Job ended at: $(date)" >> $RESOURCE_LOG
}
trap cleanup EXIT

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

echo "Loaded modules:"
module list

# Create venv
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
mkdir -p $PIP_CACHE_DIR

pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio transformers
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index hydra-core omegaconf accelerate tqdm Pillow scikit-learn
pip install seqeval

export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

echo "Starting FUNSD training with BASE model + EMA..."

# BASE model strategy:
# - Smaller model (125M vs 368M params) less prone to overfitting
# - Higher learning rate since fewer parameters
# - Larger batch size possible
# - EMA + LLRD + label smoothing
srun python scripts/train.py \
    model=layoutlmv3 \
    model.pretrained_name=microsoft/layoutlmv3-base \
    dataset=funsd \
    training.epochs=300 \
    training.batch_size=8 \
    training.gradient_accumulation_steps=4 \
    training.optimizer.learning_rate=7e-5 \
    training.optimizer.weight_decay=0.01 \
    training.scheduler.warmup_ratio=0.1 \
    +training.optimizer.llrd_rate=0.9 \
    +training.label_smoothing=0.1 \
    +training.use_ema=true \
    +training.ema_decay=0.9995 \
    hardware.mixed_precision=true \
    hardware.num_workers=8 \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    experiment.name=funsd_base_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/funsd_base_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
