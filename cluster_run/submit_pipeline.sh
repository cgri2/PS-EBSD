#!/bin/bash
set -euo pipefail

# Usage:
#   ./submit_pipeline.sh \
#     --config PATH_TO_CONFIG \
#     --start-from PART_TO_START_ON \
#     --end-on PART_TO_END_ON \
# Allowed pipeline parts:
#   1A = Pattern processing
#   1B = Neighbor pattern averaging
#   1C = Global geometry refinement
#    2 = PS refinement / CI-WCC selection
# ---- Default inputs ----
CONFIG_PATH=""
START_FROM="1A" #default (change if code series partially ran to skip completed steps); allowed: 1A, 1B, 1C, 2
END_ON="2"	#default (change to end on earlier step); allowed: 1A, 1B, 1C, 2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2;;
    --start-from) START_FROM="$2"; shift 2;;
    --end-on) END_ON="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

case "$START_FROM" in
  1A|1B|1C|2) ;;
  *) echo "--start-from must be one of: 1A, 1B, 1C, 2" >&2; exit 2;;
esac
case "$END_ON" in
  1A|1B|1C|2) ;;
  *) echo "--end-on must be one of: 1A, 1B, 1C, 2" >&2; exit 2;;
esac

part_rank() {
  case "$1" in
    1A) echo 1;;
    1B) echo 2;;
    1C) echo 3;;
    2)  echo 4;;
    *) echo "Invalid part: $1" >&2; exit 2;;
  esac
}

if (( $(part_rank "$START_FROM") > $(part_rank "$END_ON") )); then
  echo "--start-from (${START_FROM}) must be earlier than or equal to --end-on (${END_ON})" >&2
  exit 2
fi
PIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_DIR="${PIPE_ROOT}/cluster_run"

# Common exports passed to every job:
EXPORTS="ALL,CONFIG_PATH=${CONFIG_PATH},PIPE_ROOT=${PIPE_ROOT}"
CONFIG_DIR="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)"
LOG_DIR="${CONFIG_DIR}/logs"
mkdir -p "${LOG_DIR}"

dep=""  # dependency string for the next job (e.g. afterok:12345)

start_rank=$(part_rank "$START_FROM")
end_rank=$(part_rank "$END_ON")
jid1="skipped"
jid2="skipped"
jid3="skipped"
jid4="skipped"

if (( start_rank <= 1 && end_rank >= 1 )); then
  jid1=$(sbatch --parsable \
    --output="${LOG_DIR}/Part1A_%j.out" \
    --error="${LOG_DIR}/Part1A_%j.err" \
    --export="${EXPORTS}" \
    "${SBATCH_DIR}/run_Part1A.sh")
  echo "Submitted Part1A as ${jid1}"
  dep="afterok:${jid1}"
fi

if (( start_rank <= 2 && end_rank >= 2 )); then
  if [[ -n "$dep" ]]; then
    jid2=$(sbatch --parsable \
      --dependency="$dep" \
      --output="${LOG_DIR}/Part1B_%j.out" \
      --error="${LOG_DIR}/Part1B_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part1B_MPI.sh")
  else
    jid2=$(sbatch --parsable \
      --output="${LOG_DIR}/Part1B_%j.out" \
      --error="${LOG_DIR}/Part1B_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part1B_MPI.sh")
  fi
  echo "Submitted Part1B as ${jid2}${dep:+ (depends on $dep)}"
  dep="afterok:${jid2}"
fi

if (( start_rank <= 3 && end_rank >= 3 )); then
  if [[ -n "$dep" ]]; then
    jid3=$(sbatch --parsable \
      --dependency="$dep" \
      --output="${LOG_DIR}/Part1C_%j.out" \
      --error="${LOG_DIR}/Part1C_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part1C.sh")
  else
    jid3=$(sbatch --parsable \
      --output="${LOG_DIR}/Part1C_%j.out" \
      --error="${LOG_DIR}/Part1C_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part1C.sh")
  fi
  echo "Submitted Part1C as ${jid3}${dep:+ (depends on $dep)}"
  dep="afterok:${jid3}"
fi

if (( start_rank <= 4 && end_rank >= 4 )); then
  if [[ -n "$dep" ]]; then
    jid4=$(sbatch --parsable \
      --dependency="$dep" \
      --output="${LOG_DIR}/Part2_%j.out" \
      --error="${LOG_DIR}/Part2_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part2_MPI.sh")
  else
    jid4=$(sbatch --parsable \
      --output="${LOG_DIR}/Part2_%j.out" \
      --error="${LOG_DIR}/Part2_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part2_MPI.sh")
  fi
  echo "Submitted Part2 as ${jid4}${dep:+ (depends on $dep)}"
  dep="afterok:${jid4}"
fi

echo "Pipeline submitted:"
echo "  1A: ${jid1}"
echo "  1B: ${jid2}"
echo "  1C: ${jid3}"
echo "   2: ${jid4}"
