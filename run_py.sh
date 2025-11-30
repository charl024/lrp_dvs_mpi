#!/bin/bash
#SBATCH --job-name=EB_Test_single
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=./out/EB_test_single_%j.out
#SBATCH --error=./out/EB_test_single_%j.err

module load miniconda3

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dv

python ./src/main_serial.py
