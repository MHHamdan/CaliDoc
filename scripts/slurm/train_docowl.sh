#!/bin/bash
#SBATCH --job-name=calidoc_docowl
#SBATCH --account=def-YOUR_ACCOUNT  # Replace with your allocation
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2           # Need 2x H100 for 7B model
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL      # Replace with your email

# ============================================
# CaliDoc Training Script for Nibi Cluster
# Model: DocOwl 1.5 (7B)
# Dataset: DocVQA
# GPU: 2x NVIDIA H100 80GB
# ============================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load python/3.10
module load cuda/12.0

# Activate virtual environment
source ~/envs/calidoc/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_PROJECT=calidoc
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache

# For multi-GPU training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

cd $SLURM_SUBMIT_DIR/..

echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Run training with distributed data parallel
echo "Starting DocOwl training..."
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT scripts/train.py \
    experiment=main_results \
    model=docowl \
    dataset=docvqa \
    training.epochs=10 \
    training.batch_size=2 \
    training.gradient_accumulation_steps=16 \
    training.optimizer.learning_rate=2e-5 \
    model.peft.use_lora=true \
    model.peft.lora_r=16 \
    calibration.enabled=true \
    routing.enabled=true \
    hardware.mixed_precision=true \
    logging.use_wandb=true \
    experiment.output_dir=/scratch/$USER/calidoc_outputs/docowl_${SLURM_JOB_ID}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "=========================================="
