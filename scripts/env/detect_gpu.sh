#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Detect a candidate GPU in a shared system.
# Strategy:
#   - query all visible GPUs through nvidia-smi
#   - score them by low memory.used, low utilization.gpu, low power.draw
#   - prefer GPUs with utilization.gpu == 0 and very small memory.used
#
# Output:
#   By default: prints the selected GPU index to stdout
#   With --verbose: prints the scored table
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

VERBOSE=0
CSV_OUTPUT=0

for arg in "$@"; do
  case "${arg}" in
    --verbose) VERBOSE=1 ;;
    --csv) CSV_OUTPUT=1 ;;
    *)
      echo "[detect_gpu] Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[detect_gpu] ERROR: nvidia-smi not found in PATH." >&2
  exit 1
fi

QUERY_FIELDS="index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,pstate"
RAW="$(
  nvidia-smi \
    --query-gpu=${QUERY_FIELDS} \
    --format=csv,noheader,nounits
)"

if [[ -z "${RAW}" ]]; then
  echo "[detect_gpu] ERROR: no GPUs reported by nvidia-smi." >&2
  exit 1
fi

# Build a scored table: smaller is better
# Score = memory.used*1000 + util.gpu*100 + power.draw
# Strongly prefer free/idle cards.
SCORED="$(
  echo "${RAW}" | awk -F',' '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    {
      idx=trim($1)
      uuid=trim($2)
      name=trim($3)
      mem_used=trim($4)+0
      mem_total=trim($5)+0
      util_gpu=trim($6)+0
      util_mem=trim($7)+0
      power=trim($8)+0
      pstate=trim($9)

      score = mem_used*1000 + util_gpu*100 + power

      # strong bias toward completely idle cards
      if (util_gpu == 0 && mem_used <= 64) score -= 100000

      printf "%d,%s,%s,%d,%d,%d,%d,%.1f,%s,%.1f\n",
             idx, uuid, name, mem_used, mem_total, util_gpu, util_mem, power, pstate, score
    }
  ' | sort -t',' -k10,10n
)"

if [[ "${CSV_OUTPUT}" -eq 1 ]]; then
  echo "index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,pstate,score"
  echo "${SCORED}"
  exit 0
fi

if [[ "${VERBOSE}" -eq 1 ]]; then
  echo "[detect_gpu] Candidate GPUs sorted by score (smaller is better):"
  echo "index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,pstate,score"
  echo "${SCORED}"
fi

BEST_INDEX="$(echo "${SCORED}" | head -n1 | cut -d',' -f1)"

if [[ -z "${BEST_INDEX}" ]]; then
  echo "[detect_gpu] ERROR: unable to select a GPU." >&2
  exit 1
fi

echo "${BEST_INDEX}"