#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=docvqa_easyocr
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/docvqa_easyocr_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/docvqa_easyocr_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# CaliDoc - DocVQA OCR Preprocessing using EasyOCR
# Runs EasyOCR with GPU acceleration for faster processing

echo "==========================================="
echo "CaliDoc DocVQA OCR Preprocessing (EasyOCR)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "==========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs
mkdir -p /scratch/$USER/calidoc/data/docvqa

# Resource monitoring
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/docvqa_easyocr_${SLURM_JOB_ID}.log"
(
    while true; do
        echo "========== $(date) ==========" >> $RESOURCE_LOG
        nvidia-smi >> $RESOURCE_LOG 2>&1
        free -h >> $RESOURCE_LOG
        sleep 600
    done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# Load modules - opencv needed for EasyOCR
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11
module load opencv/4.10.0

echo "Loaded modules:"
module list

# Verify OpenCV works
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" || { echo "OpenCV failed"; exit 1; }

# Create venv with system site packages (keeps opencv)
python -m venv $SLURM_TMPDIR/env --system-site-packages
source $SLURM_TMPDIR/env/bin/activate

# Verify OpenCV still works
python -c "import cv2; print(f'OpenCV in venv: {cv2.__version__}')" || { echo "OpenCV in venv failed"; exit 1; }

echo "Python: $(which python)"

# Install dependencies
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index tqdm Pillow numpy scikit-image
pip install --no-index huggingface_hub requests filelock pyyaml

# EasyOCR needs to be installed with internet for models
# But the main deps should be from index
pip install easyocr --no-deps
pip install pyclipper ninja python-bidi

# Environment
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets

# EasyOCR model storage - MUST be on scratch, not home (quota issues)
export EASYOCR_MODULE_PATH=/scratch/$USER/easyocr_models
export EASYOCR_MODEL_DIR=/scratch/$USER/easyocr_models
mkdir -p $EASYOCR_MODULE_PATH

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

# Test imports
python -c "import easyocr; print('EasyOCR OK')" || echo "EasyOCR import failed"
python -c "from datasets import load_dataset; print('datasets OK')"

# Run preprocessing
SPLIT=${1:-"validation"}
echo "Starting DocVQA OCR preprocessing for split: $SPLIT..."
srun python scripts/preprocess_docvqa_streaming.py \
    --output_dir /scratch/$USER/calidoc/data/docvqa \
    --split $SPLIT \
    --save_every 500

echo "==========================================="
echo "Completed at: $(date)"
echo "==========================================="

ls -la /scratch/$USER/calidoc/data/docvqa/
