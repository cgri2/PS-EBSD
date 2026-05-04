#!/bin/bash
#SBATCH --job-name=reidx-wcc
#SBATCH --time=48:00:00
#SBATCH --ntasks=50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15G

set -euo pipefail

# Load modules
module load stack/2024-06 python/3.12.8 openmpi/4.1.6
if [[ -f "${PIPE_ROOT}/psebsd_env/bin/activate" ]]; then
  source "${PIPE_ROOT}/psebsd_env/bin/activate"
fi

# Run script
PY="${PIPE_ROOT}/ReindexingPS_Part2_NCCrefPScheck.py"
srun python -u "${PY}"

