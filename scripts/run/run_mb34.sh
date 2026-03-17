#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run MB3.4 memory benchmarks:
#   - MB3.4-A stream triad
#   - MB3.4-B pointer chasing
#
# Usage:
#   bash scripts/run/run_mb34.sh --build
#   bash scripts/run/run_mb34.sh --build --n 67108864 --iters 1048576
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0
N=67108864
BLOCK=256
GRID=0
WARMUP=5
REPS=20

WORKING_SET=1048576
ITERS=1048576
SEED=12345

OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --block) BLOCK="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --working-set) WORKING_SET="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *)
      echo "[run_mb34] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RAW_RESULTS_DIR}/mb34_${RUN_TAG}"
fi

mkdir -p "${OUTPUT_DIR}"

TRIAD_CSV="${OUTPUT_DIR}/mb34_stream_triad.csv"
PTR_CSV="${OUTPUT_DIR}/mb34_ptr_chase.csv"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/build_one.sh" mb34_stream_triad
  bash "${SCRIPT_DIR}/build_one.sh" mb34_ptr_chase
fi

echo "[run_mb34] Output dir: ${OUTPUT_DIR}"
echo "[run_mb34] Running MB3.4-A (stream triad)..."
bash "${SCRIPT_DIR}/run_one.sh" \
  --target mb34_stream_triad \
  -- \
  --device "${DEVICE}" \
  --n "${N}" \
  --block "${BLOCK}" \
  --grid "${GRID}" \
  --warmup "${WARMUP}" \
  --reps "${REPS}" \
  --csv "${TRIAD_CSV}"

echo "[run_mb34] Running MB3.4-B (pointer chasing)..."
bash "${SCRIPT_DIR}/run_one.sh" \
  --target mb34_ptr_chase \
  -- \
  --device "${DEVICE}" \
  --working-set "${WORKING_SET}" \
  --iters "${ITERS}" \
  --seed "${SEED}" \
  --csv "${PTR_CSV}"

echo "[run_mb34] Done."
echo "[run_mb34] Files:"
echo "  ${TRIAD_CSV}"
echo "  ${PTR_CSV}"