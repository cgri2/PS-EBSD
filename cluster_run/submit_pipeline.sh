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
#   2A = PS orientation refinement (one job per variant, run in parallel)
#   2B = CI/WCC selection and final map assembly (depends on all 2A jobs)
# ---- Default inputs ----
CONFIG_PATH=""
START_FROM="1A" #default (change if code series partially ran to skip completed steps); allowed: 1A, 1B, 1C, 2A, 2B
END_ON="2B"	#default (change to end on earlier step); allowed: 1A, 1B, 1C, 2A, 2B

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2;;
    --start-from) START_FROM="$2"; shift 2;;
    --end-on) END_ON="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Missing required arg: --config" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"

case "$START_FROM" in
  1A|1B|1C|2A|2B) ;;
  *) echo "--start-from must be one of: 1A, 1B, 1C, 2A, 2B" >&2; exit 2;;
esac
case "$END_ON" in
  1A|1B|1C|2A|2B) ;;
  *) echo "--end-on must be one of: 1A, 1B, 1C, 2A, 2B" >&2; exit 2;;
esac

part_rank() {
  case "$1" in
    1A) echo 1;;
    1B) echo 2;;
    1C) echo 3;;
    2A) echo 4;;
    2B) echo 5;;
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

get_resource() {
  local part="$1"
  local key="$2"
  local default="$3"

  python - <<PY
import tomllib
from pathlib import Path

config_path = Path("${CONFIG_PATH}").resolve()

with config_path.open("rb") as f:
    cfg = tomllib.load(f)

value = (
    cfg.get("resources", {})
       .get("${part}", {})
       .get("${key}", "${default}")
)

print(value)
PY
}

get_num_variants() {
  python - <<PY
import tomllib
from pathlib import Path

config_path = Path("${CONFIG_PATH}").resolve()
with config_path.open("rb") as f:
    cfg = tomllib.load(f)
ps_rotations = cfg.get("global", {}).get("PS_rotations", [])
print(len(ps_rotations) + 1)
PY
}

print_resources() {
  local part="$1"
  local time="$2"
  local ntasks="$3"
  local nodes="$4"
  local cpus="$5"
  local mem="$6"
  local partition="${7:-}"
  local account="${8:-}"
  local stagger="${9:-0}"

  echo "Submitting ${part} with resources:"
  [[ -n "${partition}" ]] && echo "  partition     = ${partition}" || echo "  partition     = (auto)"
  [[ -n "${account}"   ]] && echo "  account       = ${account}"
  echo "  time          = ${time}"
  echo "  ntasks        = ${ntasks}"
  echo "  nodes         = ${nodes}"
  echo "  cpus/task     = ${cpus}"
  echo "  mem/cpu       = ${mem}"
  [[ "${stagger}" -gt 0 ]] && echo "  stagger       = ${stagger} min between variants"
  echo ""
}

dep=""  # dependency string for the next job in the sequential chain
dep_2a=""  # dependency string collecting all Part2A job IDs for Part2B

start_rank=$(part_rank "$START_FROM")
end_rank=$(part_rank "$END_ON")
jid1="skipped"
jid2="skipped"
jid3="skipped"
jids_2a=()
jid5="skipped"

echo "Submitting pipeline:"
echo "  CONFIG_PATH = ${CONFIG_PATH}"
echo "  PIPE_ROOT   = ${PIPE_ROOT}"
echo "  SBATCH_DIR  = ${SBATCH_DIR}"
echo "  LOG_DIR     = ${LOG_DIR}"
echo ""

P1A_TIME=$(get_resource "Part1A" "time" "8:00:00")
P1A_NTASKS=$(get_resource "Part1A" "ntasks" "1")
P1A_NODES=$(get_resource "Part1A" "nodes" "1")
P1A_CPUS=$(get_resource "Part1A" "cpus_per_task" "1")
P1A_MEM=$(get_resource "Part1A" "mem_per_cpu" "64G")
P1A_PART=$(get_resource "Part1A" "partition" "normal.24h")
P1A_PART_ARG=(); [[ -n "${P1A_PART}" ]] && P1A_PART_ARG=(--partition="${P1A_PART}")
P1A_ACCT=$(get_resource "Part1A" "account" "")
P1A_ACCT_ARG=(); [[ -n "${P1A_ACCT}" ]] && P1A_ACCT_ARG=(--account="${P1A_ACCT}")

if (( start_rank <= 1 && end_rank >= 1 )); then
  print_resources "Part1A" "${P1A_TIME}" "${P1A_NTASKS}" "${P1A_NODES}" "${P1A_CPUS}" "${P1A_MEM}" "${P1A_PART}" "${P1A_ACCT}"
  jid1=$(sbatch --parsable \
    "${P1A_ACCT_ARG[@]}" \
    "${P1A_PART_ARG[@]}" \
    --time="${P1A_TIME}" \
    --ntasks="${P1A_NTASKS}" \
    --nodes="${P1A_NODES}" \
    --cpus-per-task="${P1A_CPUS}" \
    --mem-per-cpu="${P1A_MEM}" \
    --output="${LOG_DIR}/Part1A_%j.out" \
    --error="${LOG_DIR}/Part1A_%j.err" \
    --export="${EXPORTS}" \
    "${SBATCH_DIR}/run_Part1A.sh")
  echo "Submitted Part1A as ${jid1}"
  dep="afterok:${jid1}"
fi

P1B_TIME=$(get_resource "Part1B" "time" "12:00:00")
P1B_NTASKS=$(get_resource "Part1B" "ntasks" "32")
P1B_NODES=$(get_resource "Part1B" "nodes" "1")
P1B_CPUS=$(get_resource "Part1B" "cpus_per_task" "1")
P1B_MEM=$(get_resource "Part1B" "mem_per_cpu" "6G")
P1B_PART=$(get_resource "Part1B" "partition" "normal.24h")
P1B_PART_ARG=(); [[ -n "${P1B_PART}" ]] && P1B_PART_ARG=(--partition="${P1B_PART}")
P1B_ACCT=$(get_resource "Part1B" "account" "")
P1B_ACCT_ARG=(); [[ -n "${P1B_ACCT}" ]] && P1B_ACCT_ARG=(--account="${P1B_ACCT}")

if (( start_rank <= 2 && end_rank >= 2 )); then
  print_resources "Part1B" "${P1B_TIME}" "${P1B_NTASKS}" "${P1B_NODES}" "${P1B_CPUS}" "${P1B_MEM}" "${P1B_PART}" "${P1B_ACCT}"
  if [[ -n "$dep" ]]; then
    jid2=$(sbatch --parsable \
      --dependency="$dep" \
      "${P1B_ACCT_ARG[@]}" \
      "${P1B_PART_ARG[@]}" \
      --time="${P1B_TIME}" \
      --ntasks="${P1B_NTASKS}" \
      --nodes="${P1B_NODES}" \
      --cpus-per-task="${P1B_CPUS}" \
      --mem-per-cpu="${P1B_MEM}" \
      --output="${LOG_DIR}/Part1B_%j.out" \
      --error="${LOG_DIR}/Part1B_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part1B_MPI.sh")
  else
    jid2=$(sbatch --parsable \
      "${P1B_ACCT_ARG[@]}" \
      "${P1B_PART_ARG[@]}" \
      --time="${P1B_TIME}" \
      --ntasks="${P1B_NTASKS}" \
      --nodes="${P1B_NODES}" \
      --cpus-per-task="${P1B_CPUS}" \
      --mem-per-cpu="${P1B_MEM}" \
      --output="${LOG_DIR}/Part1B_%j.out" \
      --error="${LOG_DIR}/Part1B_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part1B_MPI.sh")
  fi
  echo "Submitted Part1B as ${jid2}${dep:+ (depends on $dep)}"
  dep="afterok:${jid2}"
fi

P1C_TIME=$(get_resource "Part1C" "time" "5:00:00")
P1C_NTASKS=$(get_resource "Part1C" "ntasks" "1")
P1C_NODES=$(get_resource "Part1C" "nodes" "1")
P1C_CPUS=$(get_resource "Part1C" "cpus_per_task" "1")
P1C_MEM=$(get_resource "Part1C" "mem_per_cpu" "16G")
P1C_PART=$(get_resource "Part1C" "partition" "normal.24h")
P1C_PART_ARG=(); [[ -n "${P1C_PART}" ]] && P1C_PART_ARG=(--partition="${P1C_PART}")
P1C_ACCT=$(get_resource "Part1C" "account" "")
P1C_ACCT_ARG=(); [[ -n "${P1C_ACCT}" ]] && P1C_ACCT_ARG=(--account="${P1C_ACCT}")

if (( start_rank <= 3 && end_rank >= 3 )); then
  print_resources "Part1C" "${P1C_TIME}" "${P1C_NTASKS}" "${P1C_NODES}" "${P1C_CPUS}" "${P1C_MEM}" "${P1C_PART}" "${P1C_ACCT}"
  if [[ -n "$dep" ]]; then
    jid3=$(sbatch --parsable \
      --dependency="$dep" \
      "${P1C_ACCT_ARG[@]}" \
      "${P1C_PART_ARG[@]}" \
      --time="${P1C_TIME}" \
      --ntasks="${P1C_NTASKS}" \
      --nodes="${P1C_NODES}" \
      --cpus-per-task="${P1C_CPUS}" \
      --mem-per-cpu="${P1C_MEM}" \
      --output="${LOG_DIR}/Part1C_%j.out" \
      --error="${LOG_DIR}/Part1C_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part1C.sh")
  else
    jid3=$(sbatch --parsable \
      "${P1C_ACCT_ARG[@]}" \
      "${P1C_PART_ARG[@]}" \
      --time="${P1C_TIME}" \
      --ntasks="${P1C_NTASKS}" \
      --nodes="${P1C_NODES}" \
      --cpus-per-task="${P1C_CPUS}" \
      --mem-per-cpu="${P1C_MEM}" \
      --output="${LOG_DIR}/Part1C_%j.out" \
      --error="${LOG_DIR}/Part1C_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part1C.sh")
  fi
  echo "Submitted Part1C as ${jid3}${dep:+ (depends on $dep)}"
  dep="afterok:${jid3}"
fi

P2A_TIME=$(get_resource "Part2A" "time" "72:00:00")
P2A_NTASKS=$(get_resource "Part2A" "ntasks" "100")
P2A_NODES=$(get_resource "Part2A" "nodes" "1")
P2A_CPUS=$(get_resource "Part2A" "cpus_per_task" "1")
P2A_MEM=$(get_resource "Part2A" "mem_per_cpu" "16G")
P2A_PART=$(get_resource "Part2A" "partition" "")
P2A_PART_ARG=(); [[ -n "${P2A_PART}" ]] && P2A_PART_ARG=(--partition="${P2A_PART}")
P2A_ACCT=$(get_resource "Part2A" "account" "")
P2A_ACCT_ARG=(); [[ -n "${P2A_ACCT}" ]] && P2A_ACCT_ARG=(--account="${P2A_ACCT}")
P2A_STAGGER=$(get_resource "Part2A" "stagger_minutes" "0")

if (( start_rank <= 4 && end_rank >= 4 )); then
  N_VARIANTS=$(get_num_variants)
  print_resources "Part2A (×${N_VARIANTS} variants)" "${P2A_TIME}" "${P2A_NTASKS}" "${P2A_NODES}" "${P2A_CPUS}" "${P2A_MEM}" "${P2A_PART}" "${P2A_ACCT}" "${P2A_STAGGER}"
  jid_v0=""
  for vi in $(seq 0 $((N_VARIANTS - 1))); do
    EXPORTS_VI="${EXPORTS},VARIANT_IDX=${vi}"

    # Combine chain dependency with per-variant stagger relative to V0's start time.
    # after:JOBID+N means "start no sooner than N minutes after JOBID begins", so
    # the delay is measured from when V0 actually starts (not from submission time).
    vi_dep="${dep}"
    if [[ "${vi}" -gt 0 && "${P2A_STAGGER}" -gt 0 && -n "${jid_v0}" ]]; then
      stagger_dep="after:${jid_v0}+$((vi * P2A_STAGGER))"
      vi_dep="${vi_dep:+${vi_dep},}${stagger_dep}"
    fi

    sbatch_args=(
      --parsable
      "${P2A_ACCT_ARG[@]}"
      "${P2A_PART_ARG[@]}"
      --time="${P2A_TIME}"
      --ntasks="${P2A_NTASKS}"
      --nodes="${P2A_NODES}"
      --cpus-per-task="${P2A_CPUS}"
      --mem-per-cpu="${P2A_MEM}"
      --output="${LOG_DIR}/Part2A_V${vi}_%j.out"
      --error="${LOG_DIR}/Part2A_V${vi}_%j.err"
      --export="${EXPORTS_VI}"
    )
    [[ -n "${vi_dep}" ]] && sbatch_args+=(--dependency="${vi_dep}")

    jid_vi=$(sbatch "${sbatch_args[@]}" "${SBATCH_DIR}/run_Part2A_MPI.sh")
    [[ "${vi}" -eq 0 ]] && jid_v0="${jid_vi}"
    jids_2a+=("${jid_vi}")
    echo "Submitted Part2A V${vi} as ${jid_vi}${vi_dep:+ (depends on ${vi_dep})}"
  done

  # Build dependency string for Part2B: afterok:JID0:JID1:...:JIDN
  jids_str=$(IFS=':'; echo "${jids_2a[*]}")
  dep_2a="afterok:${jids_str}"
fi

P2B_TIME=$(get_resource "Part2B" "time" "00:30:00")
P2B_NTASKS=$(get_resource "Part2B" "ntasks" "1")
P2B_NODES=$(get_resource "Part2B" "nodes" "1")
P2B_CPUS=$(get_resource "Part2B" "cpus_per_task" "1")
P2B_MEM=$(get_resource "Part2B" "mem_per_cpu" "32G")
P2B_PART=$(get_resource "Part2B" "partition" "normal.24h")
P2B_PART_ARG=(); [[ -n "${P2B_PART}" ]] && P2B_PART_ARG=(--partition="${P2B_PART}")
P2B_ACCT=$(get_resource "Part2B" "account" "")
P2B_ACCT_ARG=(); [[ -n "${P2B_ACCT}" ]] && P2B_ACCT_ARG=(--account="${P2B_ACCT}")

if (( start_rank <= 5 && end_rank >= 5 )); then
  print_resources "Part2B" "${P2B_TIME}" "${P2B_NTASKS}" "${P2B_NODES}" "${P2B_CPUS}" "${P2B_MEM}" "${P2B_PART}" "${P2B_ACCT}"
  if [[ -n "$dep_2a" ]]; then
    jid5=$(sbatch --parsable \
      --dependency="${dep_2a}" \
      "${P2B_ACCT_ARG[@]}" \
      "${P2B_PART_ARG[@]}" \
      --time="${P2B_TIME}" \
      --ntasks="${P2B_NTASKS}" \
      --nodes="${P2B_NODES}" \
      --cpus-per-task="${P2B_CPUS}" \
      --mem-per-cpu="${P2B_MEM}" \
      --output="${LOG_DIR}/Part2B_%j.out" \
      --error="${LOG_DIR}/Part2B_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part2B.sh")
  else
    jid5=$(sbatch --parsable \
      "${P2B_ACCT_ARG[@]}" \
      "${P2B_PART_ARG[@]}" \
      --time="${P2B_TIME}" \
      --ntasks="${P2B_NTASKS}" \
      --nodes="${P2B_NODES}" \
      --cpus-per-task="${P2B_CPUS}" \
      --mem-per-cpu="${P2B_MEM}" \
      --output="${LOG_DIR}/Part2B_%j.out" \
      --error="${LOG_DIR}/Part2B_%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_DIR}/run_Part2B.sh")
  fi
  echo "Submitted Part2B as ${jid5}${dep_2a:+ (depends on $dep_2a)}"
fi

echo ""
echo "Pipeline submitted:"
echo "  1A:  ${jid1}"
echo "  1B:  ${jid2}"
echo "  1C:  ${jid3}"
for vi in "${!jids_2a[@]}"; do
  echo "  2A V${vi}: ${jids_2a[$vi]}"
done
echo "  2B:  ${jid5}"
