#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run MB3.1 (device inventory, topology, optional power trace)
#
# Usage:
#   bash scripts/run/run_mb31.sh
#   bash scripts/run/run_mb31.sh --build
#   bash scripts/run/run_mb31.sh --build --trace-seconds 5
#   bash scripts/run/run_mb31.sh --device 0 --output-dir results/raw/mb31_test
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0
TRACE_SECONDS=0
OUTPUT_DIR=""
QUIET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      DO_BUILD=1
      shift
      ;;
    --device)
      if [[ $# -lt 2 ]]; then
        echo "[run_mb31] ERROR: --device requires a value." >&2
        exit 1
      fi
      DEVICE="$2"
      shift 2
      ;;
    --trace-seconds)
      if [[ $# -lt 2 ]]; then
        echo "[run_mb31] ERROR: --trace-seconds requires a value." >&2
        exit 1
      fi
      TRACE_SECONDS="$2"
      shift 2
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "[run_mb31] ERROR: --output-dir requires a value." >&2
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    *)
      echo "[run_mb31] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RAW_RESULTS_DIR}/mb31_${RUN_TAG}"
fi

mkdir -p "${OUTPUT_DIR}"

CSV_OUT="${OUTPUT_DIR}/device_inventory.csv"
TXT_OUT="${OUTPUT_DIR}/device_inventory.txt"
TRACE_OUT="${OUTPUT_DIR}/power_trace.csv"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/build_one.sh" mb31_inventory
fi

if [[ ! -x "${BIN_DIR}/mb31_inventory" ]]; then
  echo "[run_mb31] ERROR: binary not found: ${BIN_DIR}/mb31_inventory" >&2
  echo "[run_mb31] Hint: run with --build." >&2
  exit 1
fi

echo "[run_mb31] Output dir: ${OUTPUT_DIR}"
echo "[run_mb31] Device: ${DEVICE}"
echo "[run_mb31] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# 1) Device inventory
echo "[run_mb31] Running device inventory..."
"${BIN_DIR}/mb31_inventory" \
  --device "${DEVICE}" \
  --csv "${CSV_OUT}" \
  --txt "${TXT_OUT}" \
  $([[ "${QUIET}" -eq 1 ]] && echo --quiet)

# 2) Topology capture
echo "[run_mb31] Capturing topology..."
bash "${PROJECT_ROOT}/benchmarks/device/mb31_topology.sh" "${OUTPUT_DIR}"

# 3) Optional short power trace
if [[ "${TRACE_SECONDS}" -gt 0 ]]; then
  echo "[run_mb31] Capturing short power trace (${TRACE_SECONDS}s)..."
  bash "${PROJECT_ROOT}/benchmarks/device/mb31_power_trace.sh" "${TRACE_OUT}" &
  TRACE_PID=$!

  sleep "${TRACE_SECONDS}"

  kill "${TRACE_PID}" >/dev/null 2>&1 || true
  wait "${TRACE_PID}" 2>/dev/null || true
  unset TRACE_PID

  echo "[run_mb31] Power trace saved to: ${TRACE_OUT}"
fi

echo "[run_mb31] Done."
echo "[run_mb31] Files:"
echo "  ${CSV_OUT}"
echo "  ${TXT_OUT}"
echo "  ${OUTPUT_DIR}/gpu_list.txt"
echo "  ${OUTPUT_DIR}/topo_matrix.txt"
echo "  ${OUTPUT_DIR}/nvlink_status.txt"