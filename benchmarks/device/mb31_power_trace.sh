#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# MB3.1-B: GPU telemetry trace using nvidia-smi
#
# Usage:
#   bash benchmarks/device/mb31_power_trace.sh <output_csv_path>
#
# Optional env vars:
#   POWER_SAMPLE_MS=100
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

if [[ $# -lt 1 ]]; then
  echo "[mb31_power_trace] Usage: $0 <output_csv_path>" >&2
  exit 1
fi

OUT_CSV="$1"
SAMPLE_MS="${POWER_SAMPLE_MS:-100}"

if ! [[ "${SAMPLE_MS}" =~ ^[0-9]+$ ]] || [[ "${SAMPLE_MS}" -le 0 ]]; then
  echo "[mb31_power_trace] ERROR: POWER_SAMPLE_MS must be a positive integer (ms)." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[mb31_power_trace] ERROR: nvidia-smi not found." >&2
  exit 1
fi

QUERY_FIELDS="timestamp,index,uuid,name,pstate,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,utilization.memory"

echo "[mb31_power_trace] Writing telemetry to: ${OUT_CSV}"
echo "[mb31_power_trace] Sampling period: ${SAMPLE_MS} ms"
echo "[mb31_power_trace] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "timestamp,index,uuid,name,pstate,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,utilization.memory" > "${OUT_CSV}"

exec nvidia-smi \
  --query-gpu="${QUERY_FIELDS}" \
  --format=csv,noheader \
  -lms "${SAMPLE_MS}" >> "${OUT_CSV}"
