#!/bin/bash
#SBATCH --job-name=calidoc_robustness
#SBATCH --account=def-YOUR_ACCOUNT  # Replace with your allocation
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1           # H100 80GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================
# CaliDoc Robustness Evaluation
# Runs full perturbation suite on trained model
# ============================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

mkdir -p logs

module purge
module load python/3.10
module load cuda/12.0

source ~/envs/calidoc/bin/activate

export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache

cd $SLURM_SUBMIT_DIR/..

# Model checkpoint path (set this to your trained model)
MODEL_PATH=${MODEL_PATH:-"/scratch/$USER/calidoc_outputs/best"}

echo "Evaluating model: $MODEL_PATH"

python scripts/run_robustness_suite.py \
    --model_path $MODEL_PATH \
    --dataset docvqa \
    --output_dir /scratch/$USER/calidoc_outputs/robustness_${SLURM_JOB_ID} \
    --perturbations all \
    --severity_levels 1 2 3 \
    --batch_size 16

echo ""
echo "=========================================="
echo "Robustness Evaluation Complete!"
echo "End Time: $(date)"
echo "=========================================="
