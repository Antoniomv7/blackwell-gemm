#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Nsight Compute profiling for cuBLASLt baseline GEMM.
#
# Usage:
#   bash scripts/profile/profile_mb33_baseline.sh
#   bash scripts/profile/profile_mb33_baseline.sh --dtype bf16 --m 8192 --n 8192 --k 8192
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DEVICE=0
DTYPE="fp16"
M=4096
N=4096
K=4096
OUT_DIR="${NCU_RESULTS_DIR}/mb33_baseline_${RUN_TAG}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    *)
      echo "[profile_mb33_baseline] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${OUT_DIR}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "[profile_mb33_baseline] ERROR: ncu not found." >&2
  exit 1
fi

bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb4_mb5_cublaslt_baseline

echo "[profile_mb33_baseline] Profiling cuBLASLt baseline..."
ncu \
  --set full \
  --target-processes all \
  --export "${OUT_DIR}/mb4_mb5_cublaslt_baseline" \
  "${BIN_DIR}/mb4_mb5_cublaslt_baseline" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --m "${M}" \
  --n "${N}" \
  --k "${K}" \
  --warmup 3 \
  --reps 5

echo "[profile_mb33_baseline] Done. Output dir: ${OUT_DIR}"