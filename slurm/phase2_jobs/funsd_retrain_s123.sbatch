#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=funsd_retrain_s123
#SBATCH --output=/scratch/hamdan/CaliDoc/logs/%x_%j.out
#SBATCH --error=/scratch/hamdan/CaliDoc/logs/%x_%j.err
# NO --partition, NO --mem (Trillium rules)
# Instantiate: sed -e 's/funsd_retrain_s123/funsd_retrain_s42/' -e 's/funsd/funsd/' \
#   -e 's/funsd_phase2/funsd_phase2/' -e 's/123/42/' trillium_train.sbatch > job.sbatch
set -euo pipefail
[ -f /scratch/hamdan/CaliDoc/cluster_profile.sh ] && source /scratch/hamdan/CaliDoc/cluster_profile.sh
module --force purge
# Pinned 2026-07-04 from Trillium module list (cuda is a hidden module there)
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0
source /scratch/hamdan/CaliDoc/venv/bin/activate
python -c "import torch; assert torch.cuda.is_available()"
export PYTHONUNBUFFERED=1
# Compute nodes have no internet: model/processor pre-cached here by setup
export HF_HOME=/scratch/hamdan/CaliDoc/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# $HOME is read-only on Trillium compute nodes: all caches go to scratch
export XDG_CACHE_HOME=/scratch/hamdan/CaliDoc/.cache
export MPLCONFIGDIR=/scratch/hamdan/CaliDoc/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

cd /scratch/hamdan/CaliDoc/code
EXP=funsd_retrain_s123_$(date +%Y%m%d)_${SLURM_JOB_ID}
python scripts/train.py \
  dataset=funsd \
  training=funsd_phase2 \
  experiment.name=funsd_retrain_s123 \
  experiment.seed=123 \
  experiment.output_dir=/scratch/hamdan/CaliDoc/outputs/${EXP} \
  paths.bundle_dir=/scratch/hamdan/CaliDoc/data/data_bundle \
  logging.use_wandb=false
