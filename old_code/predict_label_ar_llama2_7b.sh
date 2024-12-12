#!/bin/bash

#SBATCH --job-name=predict_label_ar_llama2_7b
#SBATCH --output=/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/predict_label_ar_llama2_7b_%j.out
#SBATCH --error=/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/predict_label_ar_llama2_7b_%j.error
#SBATCH --time=23:00:00
#SBATCH --mem=64gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

OUTPUT_DIR="/fs/clip-scratch/eloirghi/CMSC828J_final_project/output"
CACHE_DIR="/fs/clip-scratch/eloirghi/CMSC828J_final_project/.cache"

source /fs/clip-scratch/eloirghi/CMSC828J_final_project/.venv/bin/activate

python -u predict_label_ar_llama2_7b.py \
  --model_name_hf "meta-llama/Llama-2-7b-chat-hf" \
  --output_path "${OUTPUT_DIR}/predict_label_ar_llama2_7b_${SLURM_JOB_ID}.jsonl" \
  --cache_dir "${CACHE_DIR}" \
  --split "test"