#!/bin/bash
#SBATCH --job-name=translate_to_darija         # Job name
#SBATCH --partition=class                   # Partition name
#SBATCH --account=class                     # Account name
#SBATCH --qos=high                          # QoS level (high priority)
#SBATCH --gres=gpu:1                        # Request one GPU
#SBATCH --cpus-per-task=8                   # Number of CPU cores
#SBATCH --mem=127G                          # Memory allocation
#SBATCH --time=20:00:00                     # Maximum runtime
#SBATCH --output=translate_to_darija_%j.out   # Main output file
#SBATCH --error=translate_to_darija_%j.err    # Main error file


# Set environment variables for output and cache directories
OUTPUT_DIR="/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/output"
CACHE_DIR="/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/.cache"

# Activate your Python environment
module load cuda
source /fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/.venv/bin/activate

# Run the Python translation script
python -u translate_ar_to_dar_llama2_7b.py \
    --model_name_hf "meta-llama/Llama-2-7b-chat-hf" \
    --output_path "${OUTPUT_DIR}/translate_ar_to_dar_llama2_7b_${SLURM_JOB_ID}.jsonl" \
    --cache_dir "${CACHE_DIR}" \
    --split "train" \
    --target_language "Moroccan Arabic"