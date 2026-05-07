#!/bin/bash
#SBATCH --job-name=npa-mpi
#SBATCH --time=05:00:00
#SBATCH --ntasks=15
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

set -euo pipefail

# Load modules and activate virtual environment
if command -v module >/dev/null 2>&1; then
  module load stack/2024-06 python/3.12.8 openmpi/4.1.6
fi

if [[ -f "${PIPE_ROOT}/psebsd_env/bin/activate" ]]; then
  source "${PIPE_ROOT}/psebsd_env/bin/activate"
fi

# Run script
PY="${PIPE_ROOT}/ReindexingPS_Part1B_NPA_MPI.py"
srun python -u "${PY}"
