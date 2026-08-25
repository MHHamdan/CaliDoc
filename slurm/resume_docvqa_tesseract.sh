#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=docvqa_tess
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/docvqa_tesseract_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/docvqa_tesseract_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# CaliDoc - DocVQA OCR Preprocessing using Tesseract
# Alternative to EasyOCR since it cannot be installed on Alliance Canada
# Uses Tesseract module which is available on the cluster

echo "==========================================="
echo "CaliDoc DocVQA OCR Preprocessing (Tesseract)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "==========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs
mkdir -p /scratch/$USER/calidoc/data/docvqa_tesseract

# Resource monitoring
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/docvqa_tesseract_${SLURM_JOB_ID}.log"
(
    echo "=== CPU/Memory Resource Monitoring ===" > $RESOURCE_LOG
    echo "Job ID: ${SLURM_JOB_ID}" >> $RESOURCE_LOG
    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG
        free -h >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG
        sleep 300
    done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# Load modules - Tesseract is available as a module!
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 python/3.11
module load tesseract/5.3.3

echo "Loaded modules:"
module list

# Verify Tesseract is available
echo "Testing Tesseract..."
tesseract --version
if [ $? -ne 0 ]; then
    echo "ERROR: Tesseract not available"
    exit 1
fi

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
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index tqdm Pillow numpy
pip install --no-index huggingface_hub requests filelock pyyaml

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets

echo "Environment setup complete."

cd /scratch/$USER/calidoc

# Check for existing checkpoint
CHECKPOINT_FILE="/scratch/$USER/calidoc/data/docvqa_tesseract/train/_checkpoint.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Found checkpoint file:"
    cat "$CHECKPOINT_FILE"
    echo ""
fi

echo "Starting DocVQA OCR preprocessing with Tesseract..."

# Run preprocessing script
srun python scripts/preprocess_docvqa_tesseract.py \
    --output_dir /scratch/$USER/calidoc/data/docvqa_tesseract \
    --batch_size 500 \
    --save_every 500

echo "==========================================="
echo "DocVQA OCR preprocessing completed at: $(date)"
echo "==========================================="

# Show final statistics
echo "Output files:"
ls -la /scratch/$USER/calidoc/data/docvqa_tesseract/
