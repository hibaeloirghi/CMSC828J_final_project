#!/bin/bash
#SBATCH --job-name=translate_eng_to_dar
#SBATCH --output=/nfshomes/eloirghi/CMSC828J_final_project/output/translate_eng_to_dar.out
#SBATCH --error=/nfshomes/eloirghi/CMSC828J_final_project/output/translate_eng_to_dar.error
#SBATCH --time=10:00:00
#SBATCH --mem=32gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1

# Set environment variables for output and cache directories
OUTPUT_DIR="/nfshomes/eloirghi/CMSC828J_final_project/output"
CACHE_DIR="/nfshomes/eloirghi/CMSC828J_final_project/.cache"

# Activate your Python environment (uncomment and adjust if necessary)
# source /path/to/your/venv/bin/activate

# Run the Python translation script
python -u translate_eng_to_dar.py \
    --model_name_hf "Helsinki-NLP/opus-mt-en-ar" \
    --output_path "${OUTPUT_DIR}/translate_eng_to_dar.jsonl" \
    --cache_dir "${CACHE_DIR}" \
    --split "train" \
    --target_language "Moroccan Darija"