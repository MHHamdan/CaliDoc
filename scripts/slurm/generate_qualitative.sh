#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --job-name=qualitative_viz
#SBATCH --output=/scratch/%u/calidoc/outputs/logs/qualitative_viz_%j.out
#SBATCH --error=/scratch/%u/calidoc/outputs/logs/qualitative_viz_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# CaliDoc - Generate Qualitative Prediction Visualizations

echo "==========================================="
echo "CaliDoc Qualitative Predictions"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started at: $(date)"
echo "==========================================="

mkdir -p /scratch/$USER/calidoc/outputs/logs

# Load modules
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/17 cuda/12.2 python/3.11

# Create virtual environment
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Set pip cache
export PIP_CACHE_DIR=/scratch/$USER/.pip_cache
export TMPDIR=$SLURM_TMPDIR
mkdir -p $PIP_CACHE_DIR

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio transformers
pip install --no-deps datasets
pip install --no-index multiprocess dill xxhash fsspec aiohttp httpx
pip install --no-index hydra-core omegaconf accelerate tqdm Pillow scikit-learn
pip install --no-index huggingface_hub requests filelock pyyaml
pip install seqeval
pip install --no-index matplotlib

# Environment variables
export PYTHONPATH=/scratch/$USER/calidoc:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
mkdir -p $MPLCONFIGDIR

cd /scratch/$USER/calidoc

srun python scripts/generate_qualitative_predictions.py \
    --funsd_checkpoint /scratch/$USER/calidoc/outputs/funsd_aug_6631427/best \
    --cord_checkpoint /scratch/$USER/calidoc/outputs/cord_optimized_6530869/best \
    --data_root /scratch/$USER/calidoc/data \
    --output_dir /scratch/$USER/calidoc/docs/paper/figures \
    --device cuda

echo "==========================================="
echo "Completed at: $(date)"
echo "==========================================="

ls -la /scratch/$USER/calidoc/docs/paper/figures/qualitative_*
