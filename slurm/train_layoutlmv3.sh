#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=calidoc_train
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/calidoc_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/calidoc_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1

# ========================================
# CaliDoc Training on Nibi (H100 GPU)
# Virtual environment created inside job per Compute Canada guidelines
# ========================================
echo "========================================"
echo "CaliDoc Training - LayoutLMv3"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Create output directories
mkdir -p /scratch/$USER/calidoc/outputs/logs

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

# ========================================
# Create virtual environment inside job (uses fast local disk)
# ========================================
echo "Creating virtual environment in \$SLURM_TMPDIR..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# Install from requirements.txt
pip install --no-index -r /scratch/$USER/calidoc/requirements.txt

# Set environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache

# Verify setup
echo ""
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Navigate to project
cd /scratch/$USER/calidoc

# ========================================
# Run Training
# ========================================
echo "Starting training..."
echo ""

srun python scripts/train.py \
    model=layoutlmv3 \
    dataset=funsd \
    training.epochs=20 \
    training.batch_size=8 \
    training.gradient_accumulation_steps=4 \
    training.optimizer.learning_rate=5e-5 \
    hardware.mixed_precision=true \
    logging.use_wandb=false \
    paths.data_root=/scratch/$USER/calidoc/data \
    paths.cache_dir=/scratch/$USER/calidoc/cache \
    experiment.name=calidoc_${SLURM_JOB_ID} \
    experiment.output_dir=/scratch/$USER/calidoc/outputs/${SLURM_JOB_ID}

EXIT_CODE=$?

# ========================================
# Backup best checkpoint to project space
# ========================================
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Backing up checkpoints..."
    BACKUP_DIR="/project/def-mcheriet/$USER/calidoc/checkpoints_$(date +%Y%m%d)"
    mkdir -p "$BACKUP_DIR"
    cp /scratch/$USER/calidoc/outputs/${SLURM_JOB_ID}/checkpoints/*.pt "$BACKUP_DIR/" 2>/dev/null
    echo "Backup: $BACKUP_DIR"
fi

echo ""
echo "========================================"
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"
