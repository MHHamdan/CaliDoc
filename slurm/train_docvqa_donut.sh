#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=donut_vqa
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/donut_vqa_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/donut_vqa_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1

# Donut DocVQA Training Script
# Generative approach for Document Visual Question Answering
# Target: ANLS >= 0.80 (SOTA Donut: ~0.80-0.83)

echo "=========================================="
echo "Donut DocVQA Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs

# Resource monitoring
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/donut_vqa_${SLURM_JOB_ID}.log"
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
pip install --no-index sentencepiece  # For Donut tokenizer

# For ANLS metric
pip install python-Levenshtein 2>/dev/null || echo "Levenshtein not available, using fallback"

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache/transformers
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /scratch/$USER/calidoc

echo "Starting Donut DocVQA training..."
nvidia-smi

# Donut training settings
# - Use pre-finetuned Donut model on DocVQA
# - Small batch due to large image size (1280x960)
# - Gradient accumulation for effective batch size 16
# - Lower LR for fine-tuning

srun python scripts/train_donut_docvqa.py \
    --model_name naver-clova-ix/donut-base-finetuned-docvqa \
    --data_dir /scratch/$USER/calidoc/data/docvqa \
    --output_dir /scratch/$USER/calidoc/outputs/donut_docvqa_${SLURM_JOB_ID} \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --max_length 128 \
    --num_workers 8 \
    --mixed_precision \
    --save_every 1 \
    --eval_every 1

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
