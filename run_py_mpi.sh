#!/bin/bash
#SBATCH --job-name=EB_Test
#SBATCH --partition=general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --output=./out/EB_test_%j.out
#SBATCH --error=./out/EB_test_%j.err

module load miniconda3
module load openmpi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dv

mpirun -n 64 python ./src/main_parallel.py
