#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run one benchmark while recording GPU telemetry in parallel.
#
# Usage:
#   bash scripts/run/run_with_trace.sh --target mb34_stream_triad --build -- --n 100000000
#
# Output:
#   - benchmark stdout/stderr -> terminal
#   - telemetry CSV -> results/power/<target>_<RUN_TAG>.csv
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

TARGET=""
DO_BUILD=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      if [[ $# -lt 2 ]]; then
        echo "[run_with_trace] ERROR: --target requires a value." >&2
        exit 1
      fi
      TARGET="$2"
      shift 2
      ;;
    --build)
      DO_BUILD=1
      shift
      ;;
    --)
      shift
      FORWARD_ARGS=("$@")
      break
      ;;
    *)
      echo "[run_with_trace] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TARGET}" ]]; then
  echo "[run_with_trace] Usage: $0 --target <name> [--build] [-- <benchmark args>]" >&2
  exit 1
fi

TRACE_SCRIPT="${PROJECT_ROOT}/benchmarks/device/mb31_power_trace.sh"
TRACE_FILE="${POWER_RESULTS_DIR}/${TARGET}_${RUN_TAG}.csv"

if [[ ! -f "${TRACE_SCRIPT}" ]]; then
  echo "[run_with_trace] ERROR: trace script not found: ${TRACE_SCRIPT}" >&2
  echo "[run_with_trace] Sprint 0/3 power trace script is required." >&2
  exit 1
fi

mkdir -p "${POWER_RESULTS_DIR}"

cleanup() {
  if [[ -n "${TRACE_PID:-}" ]]; then
    kill "${TRACE_PID}" >/dev/null 2>&1 || true
    wait "${TRACE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[run_with_trace] Starting telemetry trace -> ${TRACE_FILE}"
bash "${TRACE_SCRIPT}" "${TRACE_FILE}" &
TRACE_PID=$!

sleep 0.3

echo "[run_with_trace] Running benchmark target: ${TARGET}"
bash "${SCRIPT_DIR}/run_one.sh" \
  --target "${TARGET}" \
  $([[ "${DO_BUILD}" -eq 1 ]] && echo --build) \
  -- "${FORWARD_ARGS[@]}"

cleanup
unset TRACE_PID

echo "[run_with_trace] Trace saved to: ${TRACE_FILE}"