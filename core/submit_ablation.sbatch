#!/bin/bash
#SBATCH --job-name=pixelsmith_ablation
#SBATCH --output=results/ablation/slurm_%j.out
#SBATCH --error=results/ablation/slurm_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=jeetdevendra@umass.edu
#SBATCH --mail-type=end
# === Pixelsmith Ablation Study ===
# Full Factorial: 4 Resolutions × 3 Patch Sizes × 3 Trials = 36 runs
#
# Usage:
#   sbatch submit_ablation.sbatch              # Full ablation
#   sbatch submit_ablation.sbatch --test       # Quick test

echo "=============================================="
echo "Pixelsmith Ablation Study"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=============================================="

# GPU info
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Create output directories
mkdir -p results/images results/zoomed results/logs results/ablation

# Activate conda environment (adjust as needed)
# source activate pixelsmith

# Check Python and PyTorch
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo ""

# Parse arguments
TEST_FLAG=""
if [[ "$1" == "--test" ]]; then
    TEST_FLAG="--test"
    echo ">>> RUNNING IN TEST MODE <<<"
fi

# Run ablation study
echo "Starting ablation..."
python run_ablation.py $TEST_FLAG --seed 42

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
