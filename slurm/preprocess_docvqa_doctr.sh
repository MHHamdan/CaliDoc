#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=docvqa_doctr
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/docvqa_doctr_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/docvqa_doctr_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# CaliDoc - DocVQA OCR Preprocessing using docTR
# docTR is a PyTorch-based OCR that doesn't require opencv module

echo "==========================================="
echo "CaliDoc DocVQA OCR Preprocessing (docTR)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "==========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs
mkdir -p /scratch/$USER/calidoc/data/docvqa_doctr

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

echo "Loaded modules:"
module list

# Create virtual environment
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Python: $(which python)"

# Set pip cache
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index tqdm Pillow numpy
pip install --no-index huggingface_hub requests filelock pyyaml

# Install docTR - this may need internet for first time
# docTR PyTorch backend
pip install "python-doctr[torch]"

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export USE_TORCH=1

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

# Test docTR import
echo "Testing docTR import..."
python -c "from doctr.models import ocr_predictor; print('docTR OK')"

if [ $? -ne 0 ]; then
    echo "ERROR: docTR not available, trying alternative..."
    # Try installing with specific version
    pip install python-doctr==0.7.0
    python -c "from doctr.models import ocr_predictor; print('docTR OK')"
fi

echo "Starting DocVQA OCR preprocessing with docTR..."

srun python scripts/preprocess_docvqa_doctr.py \
    --output_dir /scratch/$USER/calidoc/data/docvqa_doctr \
    --batch_size 100 \
    --max_samples 1000

echo "==========================================="
echo "DocVQA OCR preprocessing completed at: $(date)"
echo "==========================================="

ls -la /scratch/$USER/calidoc/data/docvqa_doctr/
