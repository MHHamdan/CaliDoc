#!/bin/bash
# ============================================
# Setup CaliDoc Environment on Nibi Cluster
# Based on working Narval configuration
# ============================================
# IMPORTANT: Uses /scratch for everything to avoid home quota issues

echo "=========================================="
echo "CaliDoc Environment Setup for Nibi"
echo "=========================================="

# Configuration - USE SCRATCH TO AVOID HOME QUOTA
ENV_DIR="/scratch/$USER/envs"
ENV_NAME="calidoc"
PROJECT_DIR="/scratch/$USER/calidoc"

# Create directories on scratch
echo ""
echo "[1/5] Creating directories on /scratch..."
mkdir -p "$ENV_DIR"
mkdir -p /scratch/$USER/hf_cache
mkdir -p /scratch/$USER/torch_cache
mkdir -p /scratch/$USER/pip_cache
mkdir -p /scratch/$USER/calidoc/outputs/logs

# Set ALL cache directories to scratch (critical!)
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export PIP_CACHE_DIR=/scratch/$USER/pip_cache
export PYTHONUSERBASE=/scratch/$USER/python_packages
export XDG_CACHE_HOME=/scratch/$USER/.cache
mkdir -p $XDG_CACHE_HOME

# Load modules (arrow MUST be loaded before venv for datasets/pyarrow)
echo ""
echo "[2/5] Loading modules..."
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load arrow/17
module load cuda/12.2
module load python/3.11

echo "Loaded modules:"
module list

# Create virtual environment on scratch
echo ""
echo "[3/5] Creating virtual environment..."
if [ ! -d "$ENV_DIR/$ENV_NAME" ]; then
    python -m venv "$ENV_DIR/$ENV_NAME"
    echo "Created: $ENV_DIR/$ENV_NAME"
else
    echo "Exists: $ENV_DIR/$ENV_NAME"
fi

# Activate environment
source "$ENV_DIR/$ENV_NAME/bin/activate"

# Upgrade pip (use --no-index for Compute Canada wheels)
echo ""
echo "[4/5] Installing packages (using Compute Canada wheels)..."

# Use --no-index to avoid building from source (prevents swig errors)
pip install --upgrade pip --no-index --quiet

# Install PyTorch (pre-built for Compute Canada)
pip install --no-index torch torchvision torchaudio

# Install other packages (use --no-index where available, fallback to PyPI)
pip install --no-index transformers 2>/dev/null || pip install transformers

# datasets requires special handling (pyarrow comes from arrow module)
pip install --no-deps datasets
pip install dill multiprocess xxhash fsspec aiohttp httpx httpcore h11 anyio 2>/dev/null || true
pip install --no-index hydra-core omegaconf 2>/dev/null || pip install hydra-core omegaconf
pip install --no-index wandb 2>/dev/null || pip install wandb
pip install --no-index Pillow 2>/dev/null || pip install Pillow
pip install --no-index opencv-python-headless 2>/dev/null || pip install opencv-python-headless
pip install --no-index scikit-learn 2>/dev/null || pip install scikit-learn
pip install --no-index matplotlib seaborn pandas 2>/dev/null || pip install matplotlib seaborn pandas
pip install --no-index accelerate 2>/dev/null || pip install accelerate
pip install --no-index timm 2>/dev/null || pip install timm
pip install --no-index sentencepiece 2>/dev/null || pip install sentencepiece
pip install --no-index einops 2>/dev/null || pip install einops
pip install --no-index tqdm 2>/dev/null || pip install tqdm
pip install seqeval  # Required for NER metrics
pip install albumentations 2>/dev/null || pip install albumentations

# Optional: flash-attention (pre-built on Compute Canada)
echo ""
echo "Installing flash-attention (optional)..."
pip install --no-index flash-attn 2>/dev/null || echo "flash-attn not available (optional)"

# Verify installation
echo ""
echo "[5/5] Verification..."
echo "=========================================="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use this environment, run:"
echo ""
echo "  module --force purge"
echo "  module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11"
echo "  source /scratch/\$USER/envs/calidoc/bin/activate"
echo ""
echo "Or add to your job script."
echo ""
