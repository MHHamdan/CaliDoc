#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=docvqa_ocr
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/docvqa_ocr_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/docvqa_ocr_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# CaliDoc - DocVQA OCR Preprocessing (Resume)
# Continues OCR preprocessing from checkpoint
# Previous run completed ~12.8% before being paused

echo "==========================================="
echo "CaliDoc DocVQA OCR Preprocessing (Resume)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "==========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs
mkdir -p /scratch/$USER/calidoc/data/docvqa_ocr

# Resource monitoring
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/gpu_docvqa_ocr_${SLURM_JOB_ID}.log"
(
    echo "=== GPU/CPU Resource Monitoring ===" > $RESOURCE_LOG
    echo "Job ID: ${SLURM_JOB_ID}" >> $RESOURCE_LOG
    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG
        nvidia-smi >> $RESOURCE_LOG 2>&1
        free -h >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG
        sleep 300
    done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

echo "Loaded modules:"
module list

# Create virtual environment
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Python: $(which python)"

# Set pip cache to scratch
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index tqdm Pillow

# Install huggingface_hub and dependencies for datasets
pip install --no-index huggingface_hub requests filelock pyyaml

# Install EasyOCR (may need internet for first-time model download)
pip install easyocr

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

# Check for existing checkpoint
CHECKPOINT_FILE="/scratch/$USER/calidoc/data/docvqa_ocr/_checkpoint.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Found checkpoint file:"
    cat "$CHECKPOINT_FILE"
    echo ""
fi

echo "Starting DocVQA OCR preprocessing (will resume from checkpoint if exists)..."

# Run preprocessing script
srun python scripts/preprocess_docvqa_easyocr.py \
    --output_dir /scratch/$USER/calidoc/data/docvqa_ocr \
    --batch_size 50 \
    --save_every 500

echo "==========================================="
echo "DocVQA OCR preprocessing completed at: $(date)"
echo "==========================================="

# Show final statistics
echo "Output files:"
ls -la /scratch/$USER/calidoc/data/docvqa_ocr/
