#!/bin/bash

#SBATCH --job-name=translate_eng_to_dar_towerinstruct
#SBATCH --output=/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/translate_eng_to_dar_towerinstruct_%j.out
#SBATCH --error=/fs/clip-scratch/eloirghi/CMSC828J_final_project/output/translate_eng_to_dar_towerinstruct_%j.error
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

python -u translate_eng_to_dar_towerinstruct.py \
  --model_name_hf "Unbabel/TowerInstruct-7B-v0.2" \
  --output_path "${OUTPUT_DIR}/translate_eng_to_dar_towerinstruct_${SLURM_JOB_ID}.jsonl" \
  --cache_dir "${CACHE_DIR}" \
  --split "train" \
  --target_language "Moroccan Darija"
