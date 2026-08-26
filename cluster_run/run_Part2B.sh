#!/bin/bash
#SBATCH --job-name=ps-select
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load stack/2024-06 python/3.12.8
fi

if [[ -f "${PIPE_ROOT}/psebsd_env/bin/activate" ]]; then
  source "${PIPE_ROOT}/psebsd_env/bin/activate"
fi

PY="${PIPE_ROOT}/ReindexingPS_Part2B_PSselect.py"
python -u "${PY}"
