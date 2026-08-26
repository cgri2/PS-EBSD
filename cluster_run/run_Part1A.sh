#!/bin/bash
#SBATCH --job-name=pat-proc
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

set -euo pipefail

# Load modules and activate virtual environment
# only on euler
if command -v module >/dev/null 2>&1; then
  module load stack/2024-06 python/3.12.8 openmpi/4.1.6
fi
if [[ -f "${PIPE_ROOT}/psebsd_env/bin/activate" ]]; then
  source "${PIPE_ROOT}/psebsd_env/bin/activate"
fi

# Run script
PY="${PIPE_ROOT}/ReindexingPS_Part1A_ImportPatternProcessing.py"
srun python -u "${PY}"
