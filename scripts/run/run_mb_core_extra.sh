#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run extra core benchmarks:
#   - MB8 shared stride
#   - MB3 mem-compute mix
#   - MB10 AI-controlled kernel
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0
OUTPUT_DIR=""

STRIDE=1
BLOCK=256
GRID=120
WARMUP=5
REPS=20

N=16777216
FMAS_PER_LOAD=64

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --block) BLOCK="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --fmas-per-load) FMAS_PER_LOAD="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *)
      echo "[run_mb_core_extra] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RAW_RESULTS_DIR}/mb_core_extra_${RUN_TAG}"
fi

mkdir -p "${OUTPUT_DIR}"

MB8_CSV="${OUTPUT_DIR}/mb8_shared_stride.csv"
MB3_CSV="${OUTPUT_DIR}/mb3_mem_compute_mix.csv"
MB10_CSV="${OUTPUT_DIR}/mb10_ai_control.csv"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/build_one.sh" mb8_shared_stride
  bash "${SCRIPT_DIR}/build_one.sh" mb3_mem_compute_mix
  bash "${SCRIPT_DIR}/build_one.sh" mb10_ai_control
fi

echo "[run_mb_core_extra] Output dir: ${OUTPUT_DIR}"

echo "[run_mb_core_extra] Running MB8..."
bash "${SCRIPT_DIR}/run_one.sh" \
  --target mb8_shared_stride \
  -- \
  --device "${DEVICE}" \
  --stride "${STRIDE}" \
  --block "${BLOCK}" \
  --grid "${GRID}" \
  --warmup "${WARMUP}" \
  --reps "${REPS}" \
  --csv "${MB8_CSV}"

echo "[run_mb_core_extra] Running MB3..."
bash "${SCRIPT_DIR}/run_one.sh" \
  --target mb3_mem_compute_mix \
  -- \
  --device "${DEVICE}" \
  --n "${N}" \
  --block "${BLOCK}" \
  --grid "${GRID}" \
  --fmas-per-load "${FMAS_PER_LOAD}" \
  --warmup "${WARMUP}" \
  --reps "${REPS}" \
  --csv "${MB3_CSV}"

echo "[run_mb_core_extra] Running MB10..."
bash "${SCRIPT_DIR}/run_one.sh" \
  --target mb10_ai_control \
  -- \
  --device "${DEVICE}" \
  --n "${N}" \
  --block "${BLOCK}" \
  --grid "${GRID}" \
  --fmas-per-load "${FMAS_PER_LOAD}" \
  --warmup "${WARMUP}" \
  --reps "${REPS}" \
  --csv "${MB10_CSV}"

echo "[run_mb_core_extra] Done."
echo "[run_mb_core_extra] Files:"
echo "  ${MB8_CSV}"
echo "  ${MB3_CSV}"
echo "  ${MB10_CSV}"