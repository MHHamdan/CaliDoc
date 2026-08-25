#!/bin/bash
# Transfer rebuttal GPU scripts to Nibi cluster
# Usage: bash scripts/transfer_rebuttal_to_nibi.sh
#
# This transfers only the new rebuttal scripts (not the full project).
# Assumes the base project is already on nibi at /scratch/hamdan/calidoc/

set -e

NIBI_USER="hamdan"
NIBI_HOST="nibi.alliancecan.ca"
REMOTE_DIR="/scratch/${NIBI_USER}/calidoc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==========================================="
echo "CaliDoc Rebuttal Scripts → Nibi"
echo "==========================================="
echo ""

# SSH options
SSH_OPTS="-o ServerAliveInterval=60 -o ServerAliveCountMax=10"

# Files to transfer
FILES=(
    # GPU experiment scripts
    "scripts/evaluate_raps.py"
    "scripts/evaluate_mc_dropout.py"
    "scripts/evaluate_coverage_under_perturbations.py"
    # SLURM job script
    "scripts/slurm/run_gpu_rebuttal.sh"
    # Phase 1 scripts (in case needed on nibi for reference)
    "scripts/evaluate_mondrian_cp.py"
    "scripts/evaluate_entity_level_cp.py"
    "scripts/evaluate_cross_conformal.py"
    "scripts/evaluate_empty_sets.py"
)

echo "Files to transfer:"
for f in "${FILES[@]}"; do
    if [ -f "$PROJECT_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f (NOT FOUND)"
    fi
done

echo ""
echo "Transferring to ${NIBI_USER}@${NIBI_HOST}:${REMOTE_DIR}/..."
echo ""

# Create remote directories
ssh $SSH_OPTS "${NIBI_USER}@${NIBI_HOST}" \
    "mkdir -p ${REMOTE_DIR}/scripts/slurm ${REMOTE_DIR}/results"

# Transfer each file
for f in "${FILES[@]}"; do
    if [ -f "$PROJECT_DIR/$f" ]; then
        echo "  Sending $f ..."
        scp -q $SSH_OPTS "$PROJECT_DIR/$f" "${NIBI_USER}@${NIBI_HOST}:${REMOTE_DIR}/$f"
    fi
done

echo ""
echo "==========================================="
echo "Transfer Complete!"
echo "==========================================="
echo ""
echo "Next steps on Nibi:"
echo ""
echo "1. ssh ${NIBI_USER}@${NIBI_HOST}"
echo "2. cd ${REMOTE_DIR}"
echo ""
echo "3. Check model weights exist:"
echo "   ls outputs/funsd_aug_6631427/best/pytorch_model.bin"
echo "   ls outputs/cord_optimized_6530869/best/pytorch_model.bin"
echo ""
echo "4. Check GPU availability:"
echo "   sinfo -p gpu --format='%P %a %l %D %T %N' | head -20"
echo "   squeue -u \$USER"
echo ""
echo "5. Submit ALL experiments (RAPS + MC Dropout + Perturbations):"
echo "   sbatch scripts/slurm/run_gpu_rebuttal.sh"
echo ""
echo "   Or submit individually:"
echo "   sbatch scripts/slurm/run_gpu_rebuttal.sh raps"
echo "   sbatch scripts/slurm/run_gpu_rebuttal.sh mc_dropout"
echo "   sbatch scripts/slurm/run_gpu_rebuttal.sh perturbations"
echo ""
echo "6. Monitor:"
echo "   squeue -u \$USER"
echo "   tail -f outputs/logs/rebuttal_gpu_*.out"
echo ""
echo "7. Pull results back (from local machine):"
echo "   scp -r ${NIBI_USER}@${NIBI_HOST}:${REMOTE_DIR}/results/{raps,mc_dropout,coverage_under_perturbations} results/"
