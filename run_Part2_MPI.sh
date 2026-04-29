#!/bin/bash
#SBATCH --job-name=reidx-wcc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15G
#SBATCH --mail-user=cgriesbach@ethz.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs

# Load modules
module load stack/2024-06 python/3.12.8 openmpi/4.1.6

# Run script
PY="${PIPE_ROOT}/py/ReindexingPS_Part2_NCCrefPScheck.py"
srun python -u "${PY}"

