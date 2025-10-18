#!/bin/bash
#SBATCH --job-name=pixelsmith_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/pixelsmith_%j.out
#SBATCH --error=logs/pixelsmith_%j.err
#SBATCH --mail-user=jeetdevendra@umass.edu
#SBATCH --mail-type=ALL

# Activate environment and prepare directories
module load conda/latest

# Activate your conda environment (change to match your setup)
conda activate pixelsmith

# Ensure working directory is project root
cd $SLURM_SUBMIT_DIR

# Create log directories if missing
mkdir -p logs results/images results/zoomed results/logs


# Print job metadata
echo "===== Pixelsmith Experiment Job ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "GPU Info:"
nvidia-smi
echo "====================================="

# Run Python script
python -u generate_image.py \
  --prompt "a vast cosmic nebula swirling with radiant blue and purple gases, ultra detailed, cinematic lighting" \
  --negative_prompt "low quality, blurry, distorted, text artifacts" \
  --h_res 1024 \
  --w_res 1024 \
  --guidance_scale 7.5 \
  --slider 30

# End of job
echo "====================================="
echo "Job finished at: $(date)"