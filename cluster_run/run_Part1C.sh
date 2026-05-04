#!/bin/bash
#SBATCH --job-name=geom-refine
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G


# Load modules and activate virtual environment
module load stack/2024-06 python/3.12.8
if [[ -f "${PIPE_ROOT}/psebsd_env/bin/activate" ]]; then
  source "${PIPE_ROOT}/psebsd_env/bin/activate"
fi

# Run script
PY="${PIPE_ROOT}/ReindexingPS_Part1C_GlobalGeomRefine.py"
srun python -u "${PY}"
