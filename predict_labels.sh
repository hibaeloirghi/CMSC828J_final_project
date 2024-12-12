#!/bin/bash

OUTPUT_DIR="/fs/clip-scratch/eloirghi/CMSC828J_final_project/output"
CACHE_DIR="/fs/clip-scratch/eloirghi/CMSC828J_final_project/.cache"

# Define lists of models and languages
MODEL_NAMES=("meta-llama/Llama-2-7b-chat-hf" "Unbabel/TowerInstruct-7B-v0.2" "CohereForAI/aya-23-8B") #"meta-llama/Llama-2-13b-chat-hf") "meta-llama/Llama-2-7b-chat-hf"
LANGUAGES=("en" "ar")

# Iterate over all model-language combinations
for MODEL_NAME_HF in "${MODEL_NAMES[@]}"; do
    for LANGUAGE in "${LANGUAGES[@]}"; do
        JOB_NAME="predict_label_${LANGUAGE}_${MODEL_NAME_HF##*/}"
        sbatch << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=${OUTPUT_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${OUTPUT_DIR}/${JOB_NAME}_%j.error
#SBATCH --time=23:00:00
#SBATCH --mem=190gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

source /fs/clip-scratch/eloirghi/CMSC828J_final_project/.venv/bin/activate

echo "Processing model: $MODEL_NAME_HF, language: $LANGUAGE"
python -u predict_labels.py \
    --model_name_hf "$MODEL_NAME_HF" \
    --language "$LANGUAGE" \
    --output_path "${OUTPUT_DIR}/${JOB_NAME}_\${SLURM_JOB_ID}.jsonl" \
    --cache_dir "${CACHE_DIR}" \
    --split "test"
EOF
    done
done