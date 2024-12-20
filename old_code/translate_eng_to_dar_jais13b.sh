#!/bin/bash
#SBATCH --job-name=translate_eng_to_dar_jais13b
#SBATCH --output=/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/translate_eng_to_dar_jais13b_%j.out
#SBATCH --error=/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/translate_eng_to_dar_jais13b_%j.error
#SBATCH --time=23:00:00                # Increased time allocation
#SBATCH --mem=64gb                     # Ensure sufficient memory
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1          # Allocate a specific GPU

# Set environment variables for output and cache directories
OUTPUT_DIR="/fs/clip-scratch/eloirghi/CMSC828J_final_project/output"
CACHE_DIR="/fs/clip-scratch/eloirghi/CMSC828J_final_project/.cache"

# Activate your Python environment
source /fs/clip-scratch/eloirghi/CMSC828J_final_project/.venv/bin/activate

# Run the Python translation script
python -u translate_eng_to_dar_jais13b.py \
    --model_name_hf "inceptionai/jais-13b" \
    --output_path "${OUTPUT_DIR}/translate_eng_to_dar_jais13b_${SLURM_JOB_ID}.jsonl" \
    --cache_dir "${CACHE_DIR}" \
    --split "train" \
    --target_language "Moroccan Darija"