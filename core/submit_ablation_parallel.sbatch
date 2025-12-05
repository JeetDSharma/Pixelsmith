#!/bin/bash
#SBATCH --job-name=pixelsmith_parallel
#SBATCH --array=0-8                          # 9 experiments: 3 patch sizes × 3 trials
#SBATCH --output=results/ablation/parallel_%A_%a.out
#SBATCH --error=results/ablation/parallel_%A_%a.err
#SBATCH --time=02:00:00                      # 2 hours per experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080_ti:1                 # 1 RTX 2080 Ti per experiment
#SBATCH --mem=24G                            # 24GB RAM per job
#SBATCH --cpus-per-task=4                    # 4 CPU cores per job
#SBATCH --mail-user=jeetdevendra@umass.edu
#SBATCH --mail-type=END,FAIL                 # Email on completion or failure

# === Pixelsmith Parallel Ablation Study ===
# Runs 9 experiments in parallel (3 patch sizes × 3 trials)
# Each experiment gets its own GPU and runs independently
#
# Usage:
#   sbatch submit_ablation_parallel.sbatch
#
# Monitor progress:
#   squeue -u $USER
#   ls results/ablation/experiment_*.json
#
# Collect results after completion:
#   python core/collect_parallel_results.py

echo "=============================================="
echo "Pixelsmith Parallel Experiment ${SLURM_ARRAY_TASK_ID}"
echo "=============================================="
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=============================================="

# Display GPU information
echo "GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
echo ""

# Change to project root directory (use absolute path for SLURM reliability)
cd /home/jeetdevendra_umass_edu/Pixelsmith

# Create all necessary output directories
mkdir -p results/ablation results/images results/zoomed results/logs

# Load conda environment
echo "Loading conda environment..."
module load conda/latest
conda activate pixelsmith

# Verify Python environment
echo "Environment check:"
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Display experiment matrix for reference
echo "Experiment Matrix (ID -> Patch Size, Trial):"
echo "  0 -> (64, 0)   1 -> (64, 1)   2 -> (64, 2)"
echo "  3 -> (128, 0)  4 -> (128, 1)  5 -> (128, 2)" 
echo "  6 -> (256, 0)  7 -> (256, 1)  8 -> (256, 2)"
echo ""

# Run the single experiment
echo "Executing experiment ${SLURM_ARRAY_TASK_ID}..."
python core/run_single_experiment.py \
    --experiment_id ${SLURM_ARRAY_TASK_ID} \
    --seed 42 \
    --output_dir results/ablation

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ Experiment ${SLURM_ARRAY_TASK_ID} COMPLETED SUCCESSFULLY"
    echo "Finished: $(date)"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "❌ Experiment ${SLURM_ARRAY_TASK_ID} FAILED"
    echo "Finished: $(date)"
    echo "=============================================="
    exit 1
fi

# Show final GPU memory state
echo ""
echo "Final GPU memory state:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
