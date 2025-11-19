#!/bin/bash
#SBATCH --job-name=CS542_test
#SBATCH --partition=general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:10:00
#SBATCH --output=./out/CS542_test_%j.out
#SBATCH --error=./out/CS542_test_%j.err

module load miniconda3
module load openmpi

source $(conda info --base)/etc/profile.d/conda.sh

conda activate dv

mpirun -n 64 python ./src/mpi_test.py
