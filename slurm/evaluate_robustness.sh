#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=robustness_eval
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/robustness_eval_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/robustness_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# CaliDoc - Robustness Evaluation under Document Perturbations
# Evaluates model performance under:
# - Blur (Gaussian, motion)
# - Noise (Gaussian, salt-pepper)
# - Compression (JPEG)
# - Photometric (brightness, contrast)
# - Document-specific (ink bleed, paper texture)

echo "==========================================="
echo "CaliDoc Robustness Evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started at: $(date)"
echo "==========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs
mkdir -p /scratch/$USER/calidoc/results

# Load modules - IMPORTANT: opencv MUST be loaded BEFORE creating venv
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11
module load opencv/4.10.0

echo "Loaded modules:"
module list

# Verify OpenCV is available BEFORE venv
echo "Testing OpenCV import before venv..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Create virtual environment AFTER loading opencv
python -m venv $SLURM_TMPDIR/env --system-site-packages
source $SLURM_TMPDIR/env/bin/activate

# Verify OpenCV still works in venv
echo "Testing OpenCV import in venv..."
python -c "import cv2; print(f'OpenCV version in venv: {cv2.__version__}')"

echo "Python: $(which python)"

# Set pip cache to scratch
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio transformers
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index hydra-core omegaconf accelerate tqdm Pillow scikit-learn
pip install seqeval

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets

# Fix matplotlib/fontconfig warnings
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
mkdir -p $MPLCONFIGDIR

echo "Environment setup complete."
nvidia-smi

cd /scratch/$USER/calidoc

# Configuration
CHECKPOINT_PATH=${1:-"/scratch/$USER/calidoc/outputs/funsd_aug_6631427/best"}
DATASET=${2:-"funsd"}
OUTPUT_DIR="/scratch/$USER/calidoc/results/robustness_${DATASET}_${SLURM_JOB_ID}"

echo "==========================================="
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT_DIR"
echo "==========================================="

# Run robustness evaluation
echo "Starting robustness evaluation..."
srun python scripts/evaluate_conformal.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset "$DATASET" \
    --data_root /scratch/$USER/calidoc/data \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --num_workers 4 \
    --device cuda \
    --run_robustness

echo "==========================================="
echo "Evaluation completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "==========================================="

ls -la "$OUTPUT_DIR"
