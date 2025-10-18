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
#SBATCH --mail-type=end

# === Environment setup ===
module load conda/latest
conda activate pixelsmith
cd $SLURM_SUBMIT_DIR

mkdir -p logs results/images results/zoomed results/logs

# === Job metadata ===
echo "===== Pixelsmith Experiment Job ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "GPU Info:"
nvidia-smi
echo "====================================="

# === Run Python ===
python -u generate_image.py \
  --prompt "a neatly arranged tabletop scene with a glass of water, a metallic pen, a textured leather notebook, a smartphone screen showing icons, and a folded cloth napkin, all under soft studio lighting, ultra high detail, sharp focus, 8k realism"\
  --negative_prompt "blurry, overexposed, underexposed, low quality, noise, distorted edges"\
  --h_res 1024 \
  --w_res 1024 \
  --guidance_scale 9.0 \
  --max_scale 8 \
  --seed 1643170768 \
  # --slider 0.7

echo "====================================="
echo "Job finished at: $(date)"
