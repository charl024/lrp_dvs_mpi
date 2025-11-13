#!/bin/bash
#SBATCH --job-name=CS542_test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=./out/CS542_test_%j.out
#SBATCH --error=./out/CS542_test_%j.err

module load miniconda3

source $(conda info --base)/etc/profile.d/conda.sh

conda activate dv

python ./src/hist_heatmap.py
