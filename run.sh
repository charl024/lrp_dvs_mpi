#!/bin/bash
#SBATCH --job-name=EB_MPI
#SBATCH --partition=general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --output=./out_many/EB_MPI_TEST_%j.out
#SBATCH --error=./out_many/EB_MPI_TEST_%j.err

module load miniconda3
module load openmpi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dv

N=10

for ((i=1; i<=N; i++)); do
    echo "Iteration $i"

    echo "Running with 64 processes"
    mpirun -n 64 python ./src/main_parallel.py  --process-width 8 --process-height 8 --packet-size 1000

    echo "Running with 32 (8x4) processes"
    mpirun -n 32 python ./src/main_parallel.py --process-width 8 --process-height 4 --packet-size 1000

    echo "Running with 32 (4x8) processes"
    mpirun -n 32 python ./src/main_parallel.py --process-width 4 --process-height 8 --packet-size 1000

    echo "Running with 16 processes"
    mpirun -n 16 python ./src/main_parallel.py --process-width 4 --process-height 4 --packet-size 1000

    echo "Running with 4 processes"
    mpirun -n 4 python ./src/main_parallel.py --process-width 2 --process-height 2 --packet-size 1000

    echo "Running serial"
    python ./src/main_serial.py

    echo "----------------------------------------"
done

