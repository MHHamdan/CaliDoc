#!/bin/bash
#SBATCH --job-name=extract_cord
#SBATCH --output=logs/extract_cord_%j.out
#SBATCH --error=logs/extract_cord_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Extract model analysis data for CORD

module load python/3.10 cuda/12.1

cd $SLURM_TMPDIR
cp -r ~/projects/def-sponsor00/mhussein/calidoc .
cd calidoc

# Activate environment
source ~/envs/calidoc/bin/activate

# Find best checkpoint
CHECKPOINT_DIR="outputs/cord_sota/best"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_DIR=$(ls -td outputs/cord*/checkpoint-* 2>/dev/null | head -1)
fi

echo "Using checkpoint: $CHECKPOINT_DIR"

# Create output directory
mkdir -p analysis_outputs

# Run extraction
python scripts/extract_model_analysis.py \
    --task cord \
    --checkpoint $CHECKPOINT_DIR \
    --output_dir analysis_outputs \
    --num_attention_samples 20 \
    --num_qualitative_samples 30 \
    --device cuda

# Copy results back
cp -r analysis_outputs ~/projects/def-sponsor00/mhussein/calidoc/

echo "Analysis extraction complete!"
