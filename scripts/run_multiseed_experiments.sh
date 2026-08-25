#!/bin/bash
#SBATCH --job-name=calidoc_multiseed
#SBATCH --output=/scratch/hamdan/calidoc_outputs/multiseed_%A_%a.out
#SBATCH --error=/scratch/hamdan/calidoc_outputs/multiseed_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8

# Multi-seed experiments for statistical significance
# Runs 3 seeds (42, 123, 456) x 3 datasets (FUNSD, CORD, DocVQA)

SEEDS=(42 123 456)
DATASETS=("funsd" "cord" "docvqa")

# Calculate which seed and dataset to use
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 3))

SEED=${SEEDS[$SEED_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

echo "Running experiment: Dataset=$DATASET, Seed=$SEED"

# Activate environment
source /scratch/hamdan/miniconda3/bin/activate calidoc

# Set output directory
OUTPUT_DIR="/scratch/hamdan/calidoc_outputs/multiseed_${DATASET}_seed${SEED}"
mkdir -p $OUTPUT_DIR

cd /scratch/hamdan/calidoc

if [ "$DATASET" == "funsd" ]; then
    python train.py \
        --config configs/experiment/funsd_sota.yaml \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        experiment.name="multiseed_funsd_seed${SEED}"

elif [ "$DATASET" == "cord" ]; then
    python train.py \
        --config configs/experiment/cord_sota.yaml \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        experiment.name="multiseed_cord_seed${SEED}"

elif [ "$DATASET" == "docvqa" ]; then
    python train.py \
        --config configs/experiment/docvqa_pix2struct.yaml \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        experiment.name="multiseed_docvqa_seed${SEED}"
fi

echo "Experiment completed: Dataset=$DATASET, Seed=$SEED"
