#!/bin/bash
# Transfer CaliDoc project to Nibi cluster
# Usage: bash scripts/transfer_to_nibi.sh
#
# Features:
#   - Resumable transfer (rsync --partial)
#   - Keep SSH alive for large transfers
#   - Compression for faster transfer

set -e

# Configuration
NIBI_USER="hamdan"
NIBI_HOST="nibi.alliancecan.ca"
REMOTE_DIR="/scratch/${NIBI_USER}/calidoc"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "CaliDoc Transfer to Nibi Cluster"
echo "=========================================="
echo ""
echo "Source: $PROJECT_DIR"
echo "Destination: ${NIBI_USER}@${NIBI_HOST}:${REMOTE_DIR}"
echo "Total size: $(du -sh "$PROJECT_DIR" | cut -f1)"
echo ""

# SSH options to keep connection alive
SSH_OPTS="-o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes"

# Create remote directory
echo "Creating remote directory..."
ssh $SSH_OPTS "${NIBI_USER}@${NIBI_HOST}" "mkdir -p ${REMOTE_DIR}"

# Files/directories to exclude from transfer
EXCLUDE_OPTS=(
    --exclude='.git'
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.pytest_cache'
    --exclude='*.egg-info'
    --exclude='.eggs'
    --exclude='cache/'
    --exclude='wandb/'
    --exclude='.venv'
    --exclude='venv'
    --exclude='*.log'
    --exclude='.DS_Store'
)

# Transfer project with resume capability
echo ""
echo "Transferring project files (resumable)..."
echo "This may take a while for large datasets..."
echo ""

rsync -avz --progress --partial --partial-dir=.rsync-partial \
    --timeout=300 \
    -e "ssh $SSH_OPTS" \
    "${EXCLUDE_OPTS[@]}" \
    "$PROJECT_DIR/" \
    "${NIBI_USER}@${NIBI_HOST}:${REMOTE_DIR}/"

echo ""
echo "=========================================="
echo "Transfer Complete!"
echo "=========================================="
echo ""
echo "Next steps on Nibi cluster:"
echo ""
echo "1. Connect to Nibi:"
echo "   ssh ${NIBI_USER}@${NIBI_HOST}"
echo ""
echo "2. Navigate to project:"
echo "   cd ${REMOTE_DIR}"
echo ""
echo "3. Set up environment (run once):"
echo "   bash scripts/slurm/setup_environment.sh"
echo ""
echo "4. Submit training job:"
echo "   sbatch scripts/slurm/train_layoutlmv3.sh"
echo ""
