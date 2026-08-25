#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G          # MANDATORY — Nibi default 256M OOM-kills in seconds
#SBATCH --time=12:00:00
#SBATCH --job-name=__JOBNAME__
#SBATCH --output=/scratch/hamdan/CaliDoc/logs/%x_%j.out
#SBATCH --error=/scratch/hamdan/CaliDoc/logs/%x_%j.err
# Instantiate: sed -e 's/__JOBNAME__/cord_retrain_s42/' -e 's/__DATASET__/cord/' \
#   -e 's/__TRAINCFG__/cord_phase2/' -e 's/__SEED__/42/' nibi_train.sbatch > job.sbatch
set -euo pipefail
[ -f /scratch/hamdan/CaliDoc/cluster_profile.sh ] && source /scratch/hamdan/CaliDoc/cluster_profile.sh
module --force purge
# Pinned 2026-07-04 from Nibi module avail — identical stack to Trillium
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
EXP=__JOBNAME___$(date +%Y%m%d)_${SLURM_JOB_ID}
python scripts/train.py \
  dataset=__DATASET__ \
  training=__TRAINCFG__ \
  experiment.name=__JOBNAME__ \
  experiment.seed=__SEED__ \
  experiment.output_dir=/scratch/hamdan/CaliDoc/outputs/${EXP} \
  paths.bundle_dir=/scratch/hamdan/CaliDoc/data/data_bundle \
  logging.use_wandb=false
