#!/bin/bash
#SBATCH --job-name=geom-refine
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=cgriesbach@ethz.ch
#SBATCH --mail-type=END,FAIL


# Load modules and activate virtual environment
module load stack/2024-06 python/3.12.8

# Run script
PY="${PIPE_ROOT}/py/ReindexingPS_Part1C_GlobalGeomRefine.py"
srun python -u "${PY}"
