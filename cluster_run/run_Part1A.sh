#!/bin/bash
#SBATCH --job-name=pat-proc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=96G
#SBATCH --mail-user=cgriesbach@ethz.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs

# Load modules and activate virtual environment
module load stack/2024-06 python/3.12.8
source /cluster/home/cgriesbach/kp_env/bin/activate

# Run script
PY="${PIPE_ROOT}/py/ReindexingPS_Part1A_PatternProcessing.py"
srun python -u "${PY}"
