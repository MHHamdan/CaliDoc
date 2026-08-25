#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=funsd_conformal_eval
#SBATCH --output=/scratch/hamdan/CaliDoc/logs/%x_%j.out
#SBATCH --error=/scratch/hamdan/CaliDoc/logs/%x_%j.err
# NO --partition, NO --mem (Trillium rules). Submit from trig-login01.
set -euo pipefail
[ -f /scratch/hamdan/CaliDoc/cluster_profile.sh ] && source /scratch/hamdan/CaliDoc/cluster_profile.sh
module --force purge
# Pinned 2026-07-04 (identical stack to Nibi)
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0
source /scratch/hamdan/CaliDoc/venv/bin/activate
python -c "import torch; assert torch.cuda.is_available()"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/hamdan/CaliDoc/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/scratch/hamdan/CaliDoc/.cache
export MPLCONFIGDIR=/scratch/hamdan/CaliDoc/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

cd /scratch/hamdan/CaliDoc/code
# Protocol: temperature on es_holdout -> per-seed {42,123,456} calibration
# subsets of test -> mean±std; LAC + APS at 90/95/99; Mondrian per class.
python scripts/evaluate_conformal_phase2.py \
  --model_path /scratch/hamdan/CaliDoc/outputs/funsd_retrain_s42_20260704_634119/best \
  --dataset funsd \
  --bundle_dir /scratch/hamdan/CaliDoc/data/data_bundle \
  --output /scratch/hamdan/CaliDoc/results/funsd_conformal_eval.json
