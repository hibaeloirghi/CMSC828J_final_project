#!/bin/bash
#SBATCH --job-name=clean_translation         # Job name
#SBATCH --partition=class                   # Partition name
#SBATCH --account=class                     # Account name
#SBATCH --gres=gpu:1                        # Request one GPU
#SBATCH --mem=2G                          # Memory allocation
#SBATCH --time=00:01:00                     # Maximum runtime
#SBATCH --output=clean_translation_%j.out   # Main output file
#SBATCH --error=clean_translation_%j.err    # Main error file

module load cuda
source /fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/.venv/bin/activate

# Run the Python script
python /fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/output/clean_translation.py
