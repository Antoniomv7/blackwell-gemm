#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run cuBLASLt GEMM baseline (Sprint 8).
#
# Usage:
#   bash scripts/run/run_mb33_baseline.sh --build
#   bash scripts/run/run_mb33_baseline.sh --build --dtype bf16 --m 8192 --n 8192 --k 8192
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0
DTYPE="fp16"
M=4096
N=4096
K=4096
WARMUP=5
REPS=20
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *)
      echo "[run_mb33_baseline] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RAW_RESULTS_DIR}/mb33_baseline_${RUN_TAG}"
fi

mkdir -p "${OUTPUT_DIR}"

CSV_OUT="${OUTPUT_DIR}/mb4_mb5_cublaslt_baseline.csv"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/build_one.sh" mb4_mb5_cublaslt_baseline
fi

echo "[run_mb33_baseline] Output dir: ${OUTPUT_DIR}"
echo "[run_mb33_baseline] Running cuBLASLt baseline..."

bash "${SCRIPT_DIR}/run_one.sh" \
  --target mb4_mb5_cublaslt_baseline \
  -- \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --m "${M}" \
  --n "${N}" \
  --k "${K}" \
  --warmup "${WARMUP}" \
  --reps "${REPS}" \
  --csv "${CSV_OUT}"

echo "[run_mb33_baseline] Done."
echo "[run_mb33_baseline] File:"
echo "  ${CSV_OUT}"