#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run MB3.2:
#   - MB3.2-A residency / occupancy probe
#   - MB3.2-B FP32 dependency-vs-throughput probe
#
# Usage:
#   bash scripts/run/run_mb32.sh --build
#   bash scripts/run/run_mb32.sh --build --reg-footprint 64 --shared-bytes 32768
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0

REG_FOOTPRINT=32
SHARED_BYTES=0
BLOCK=256
GRID=120
WARMUP=5
REPS=20

OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --reg-footprint) REG_FOOTPRINT="$2"; shift 2 ;;
    --shared-bytes) SHARED_BYTES="$2"; shift 2 ;;
    --block) BLOCK="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *)
      echo "[run_mb32] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RAW_RESULTS_DIR}/mb32_${RUN_TAG}"
fi

mkdir -p "${OUTPUT_DIR}"

RESIDENCY_CSV="${OUTPUT_DIR}/mb32_residency.csv"
FP32_CSV="${OUTPUT_DIR}/mb32_fp32_probe.csv"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/build_one.sh" mb32_residency
  bash "${SCRIPT_DIR}/build_one.sh" mb32_fp32_probe
fi

echo "[run_mb32] Output dir: ${OUTPUT_DIR}"

echo "[run_mb32] Running MB3.2-A (residency)..."
bash "${SCRIPT_DIR}/run_one.sh" \
  --target mb32_residency \
  -- \
  --device "${DEVICE}" \
  --reg-footprint "${REG_FOOTPRINT}" \
  --shared-bytes "${SHARED_BYTES}" \
  --block "${BLOCK}" \
  --grid "${GRID}" \
  --warmup "${WARMUP}" \
  --reps "${REPS}" \
  --csv "${RESIDENCY_CSV}"

echo "[run_mb32] Running MB3.2-B (FP32 probe)..."
bash "${SCRIPT_DIR}/run_one.sh" \
  --target mb32_fp32_probe \
  -- \
  --device "${DEVICE}" \
  --block "${BLOCK}" \
  --grid "${GRID}" \
  --warmup "${WARMUP}" \
  --reps "${REPS}" \
  --csv "${FP32_CSV}"

echo "[run_mb32] Done."
echo "[run_mb32] Files:"
echo "  ${RESIDENCY_CSV}"
echo "  ${FP32_CSV}"