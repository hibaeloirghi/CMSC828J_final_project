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

module load cuda
source /fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/.venv/bin/activate

# Arrays of models and languages
MODEL_NAMES=("meta-llama/Llama-3.1-8B" "meta-llama/Llama-2-7b-chat-hf")
LANGUAGES=("en" "ar")

# Output directory
OUTPUT_DIR="/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/output"
CACHE_DIR="/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/.cache"

# Iterate over all combinations
for MODEL_NAME_HF in "${MODEL_NAMES[@]}"; do
  for LANGUAGE in "${LANGUAGES[@]}"; do
    # Create unique output and error filenames
    OUTPUT_FILE="${OUTPUT_DIR}/output_${MODEL_NAME_HF##*/}_${LANGUAGE}.out"
    ERROR_FILE="${OUTPUT_DIR}/error_${MODEL_NAME_HF##*/}_${LANGUAGE}.err"
    
    # Launch the Python script as a background job, redirecting output and error
    python translate_to_darija.py \
      --model_name_hf "$MODEL_NAME_HF" \
      --source_language "$LANGUAGE" \
      --output_path "${OUTPUT_DIR}/${MODEL_NAME_HF##*/}_${LANGUAGE}.jsonl" \
      --cache_dir "$CACHE_DIR" \
      --split "test" > "$OUTPUT_FILE" 2> "$ERROR_FILE" &

    # Limit the number of concurrent jobs
    if (( $(jobs -r | wc -l) >= 1 )); then
      wait -n
    fi
  done
done

# Wait for all background jobs to finish
wait

echo "All jobs have completed."