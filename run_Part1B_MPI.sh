#!/bin/bash
#SBATCH --job-name=npa-mpi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --ntasks=15
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=cgriesbach@ethz.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs

# Load modules and activate virtual environment
module load stack/2024-06 python/3.12.8 openmpi/4.1.6

# force single‐threaded BLAS/OpenMP
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMPI_MCA_plm=slurm

#source /cluster/home/cgriesbach/kp_env/bin/activate

# Run script
PY="${PIPE_ROOT}/py/ReindexingPS_Part1B_NPA_MPI.py"
srun python -u "${PY}"
