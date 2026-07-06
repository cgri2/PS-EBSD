#!/bin/bash
#SBATCH --job-name=ps-refine
#SBATCH --time=72:00:00
#SBATCH --ntasks=100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load stack/2024-06 python/3.12.8 openmpi/4.1.6
fi

source /cluster/work/mandm/cgriesbach/EBSDindexing/PSEBSD/PS-EBSD/psebsd_env/bin/activate

PY="${PIPE_ROOT}/ReindexingPS_Part2A_NCCref.py"
srun python -u "${PY}"
