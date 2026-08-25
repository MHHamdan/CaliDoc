#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --gpus=h100:1
# Forward-pass only => MIG-compatible. Keep --gpus=h100:1 unless the user
# confirms a MIG gres string from `scontrol show node`; MIG jobs schedule
# sooner around Nibi maintenance windows. If PD with "ReqNodeNotAvail,
# Reserved for maintenance", check `scontrol show reservations` first.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G          # MANDATORY — Nibi default 256M OOM-kills in seconds
#SBATCH --time=04:00:00
#SBATCH --job-name=__JOBNAME__
#SBATCH --output=/scratch/hamdan/CaliDoc/logs/%x_%j.out
#SBATCH --error=/scratch/hamdan/CaliDoc/logs/%x_%j.err
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
# Protocol: temperature on val_es -> quantiles on cal -> evaluate test;
# LAC + APS at 90/95/99; cross-conformal K=5 secondary (>2pp gap = red flag).
# Set __TRAIN_EXP__ to the finished retrain's output dir name before sbatch.
python scripts/evaluate_conformal_phase2.py \
  --model_path /scratch/hamdan/CaliDoc/outputs/__TRAIN_EXP__/best \
  --dataset __DATASET__ \
  --bundle_dir /scratch/hamdan/CaliDoc/data/data_bundle \
  --output /scratch/hamdan/CaliDoc/results/__JOBNAME__.json
