#!/bin/bash
#SBATCH --job-name=submit-psebsd
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100M
#SBATCH --output=submit_pipeline_%j.out
#SBATCH --error=submit_pipeline_%j.err

set -euo pipefail

EXAMPLE_DIR='/{data_filepath}'
REPO_ROOT='/{filepath_to_repo}/PS-EBSD/'
bash "${REPO_ROOT}/cluster_run/submit_pipeline.sh" \
  --config "${EXAMPLE_DIR}/PS-EBSD_config.toml" \
  --start-from 1A \
  --end-on 2B 
