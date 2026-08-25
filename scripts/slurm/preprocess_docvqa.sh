#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=docvqa_ocr
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/docvqa_ocr_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/docvqa_ocr_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# DocVQA OCR Preprocessing on Nibi Cluster
# Uses EasyOCR with GPU for 10-50x speedup over CPU

echo "=========================================="
echo "DocVQA OCR Preprocessing"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "=========================================="

# Create directories
mkdir -p /scratch/$USER/calidoc/outputs/logs
mkdir -p /scratch/$USER/calidoc/data/docvqa

# Define log files
RESOURCE_LOG="/scratch/$USER/calidoc/outputs/logs/cpu_utilizations_${SLURM_JOB_ID}.log"
echo "Resource monitoring log: $RESOURCE_LOG"

# Start resource monitoring in background (every 60 seconds)
(
    echo "=== Resource Monitoring Started ===" > $RESOURCE_LOG
    echo "Job ID: ${SLURM_JOB_ID}" >> $RESOURCE_LOG
    echo "Node: ${SLURMD_NODENAME}" >> $RESOURCE_LOG
    echo "" >> $RESOURCE_LOG

    while true; do
        echo "=== $(date) ===" >> $RESOURCE_LOG
        echo "--- CPU Usage ---" >> $RESOURCE_LOG
        top -bn1 | head -20 >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG
        echo "--- Memory Usage ---" >> $RESOURCE_LOG
        free -h >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG
        echo "--- Disk Usage ---" >> $RESOURCE_LOG
        df -h /scratch/$USER 2>/dev/null >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG
        echo "--- Process Stats ---" >> $RESOURCE_LOG
        ps aux --sort=-%cpu | head -10 >> $RESOURCE_LOG
        echo "" >> $RESOURCE_LOG
        sleep 300  # Every 5 minutes
    done
) &
MONITOR_PID=$!
echo "Started resource monitor with PID: $MONITOR_PID"

# Cleanup function
cleanup() {
    echo "Stopping resource monitor..."
    kill $MONITOR_PID 2>/dev/null
    echo "=== Final Resource Stats ===" >> $RESOURCE_LOG
    echo "Job ended at: $(date)" >> $RESOURCE_LOG
}
trap cleanup EXIT

# ============================================
# CRITICAL: Load modules BEFORE creating venv
# Arrow and opencv must be loaded before venv
# ============================================
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 opencv/4.10.0 python/3.11

echo "Checking GPU..."
nvidia-smi

echo "Loaded modules:"
module list

# Create virtual environment
echo "Creating virtual environment..."
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Python location: $(which python)"
echo "Pip location: $(which pip)"

# CRITICAL: Set pip cache to scratch to avoid home quota issues
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

# Upgrade pip
pip install --no-index --upgrade pip

# Install packages using cluster wheels (--no-index)
echo "Installing dependencies..."
pip install --no-index tqdm Pillow numpy
pip install --no-index multiprocess dill xxhash fsspec aiohttp
pip install --no-index torch torchvision

# Install datasets dependencies first (datasets was installed with --no-deps)
echo "Installing datasets dependencies..."
pip install --no-index huggingface_hub || pip install huggingface_hub
pip install --no-index requests || pip install requests
pip install --no-index httpx || pip install httpx
pip install --no-index pyyaml || pip install pyyaml
pip install --no-index filelock || pip install filelock

# Now install datasets without deps to avoid pyarrow version conflict
pip install --no-deps datasets

# EasyOCR - opencv is provided by module, install easyocr without opencv dependency
echo "Installing EasyOCR..."
pip install --no-index ninja  # Build dependency (from cluster wheels)
pip install --no-deps easyocr  # Install without deps to avoid opencv conflict

# EasyOCR dependencies - use cluster wheels where available
echo "Installing EasyOCR dependencies..."
pip install --no-index shapely  # Available in cluster
pip install --no-index scikit-image  # Available in cluster as skimage
pip install --no-index pyclipper || pip install pyclipper  # Try cluster first
pip install --no-index pyyaml || pip install pyyaml  # EasyOCR needs this

# Fix python-bidi compatibility issue
# The cluster's python-bidi (0.4.2+computecanada) doesn't have get_display
# Since we only do English OCR, we can patch the bidi module
echo "Patching bidi module for English-only OCR..."
python -c "
import sys
import types

# Create a dummy bidi module with get_display that returns text as-is
bidi_module = types.ModuleType('bidi')
def get_display(text, *args, **kwargs):
    return text  # For English, no RTL processing needed
bidi_module.get_display = get_display

# Also create bidi.algorithm submodule
algorithm_module = types.ModuleType('bidi.algorithm')
algorithm_module.get_display = get_display
bidi_module.algorithm = algorithm_module

sys.modules['bidi'] = bidi_module
sys.modules['bidi.algorithm'] = algorithm_module

print('Bidi module patched for English-only OCR')
"

# Now install easyocr's remaining deps (python-bidi will use our patch)
pip install python-bidi || true  # May fail but we have the patch

# Verify installations
echo "Verifying installations..."
python -c "
import sys
import types

# Patch bidi before importing easyocr
bidi_module = types.ModuleType('bidi')
def get_display(text, *args, **kwargs):
    return text
bidi_module.get_display = get_display
algorithm_module = types.ModuleType('bidi.algorithm')
algorithm_module.get_display = get_display
bidi_module.algorithm = algorithm_module
sys.modules['bidi'] = bidi_module
sys.modules['bidi.algorithm'] = algorithm_module

import easyocr
print('EasyOCR version:', easyocr.__version__)
" || {
    echo "ERROR: EasyOCR installation failed!"
    exit 1
}

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export EASYOCR_MODULE_PATH=/scratch/$USER/easyocr_models

echo ""
echo "Environment setup complete."
echo "PYTHONPATH: $PYTHONPATH"
echo ""

cd /scratch/$USER/calidoc

echo "Starting DocVQA OCR preprocessing with EasyOCR..."
echo ""

# Run the preprocessing script with checkpointing
# --batch_size 500: Save checkpoint every 500 samples for OOM resilience
python scripts/preprocess_docvqa_easyocr.py \
    --output_dir /scratch/$USER/calidoc/data/docvqa \
    --splits train validation \
    --batch_size 500

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
