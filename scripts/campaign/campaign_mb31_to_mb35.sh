#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Main reproducible campaign for the project (safe baseline path).
#
# Includes:
#   - MB3.1  device inventory / topology / optional trace
#   - MB3.2  residency + FP32 probe
#   - MB3.4  stream triad + pointer chasing
#   - MB8 / MB3 / MB10 core extra benchmarks
#   - cuBLASLt baseline GEMM
#
# Usage:
#   bash scripts/campaign/campaign_mb31_to_mb35.sh --build
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0
TRACE_SECONDS=3
OUTPUT_ROOT=""
DTYPE="fp16"
M=4096
N=4096
K=4096

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --trace-seconds) TRACE_SECONDS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "[campaign_mb31_to_mb35] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${RAW_RESULTS_DIR}/campaign_${RUN_TAG}"
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[campaign_mb31_to_mb35] Output root: ${OUTPUT_ROOT}"
echo "[campaign_mb31_to_mb35] Device: ${DEVICE}"
echo "[campaign_mb31_to_mb35] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# --------------------------------------------------------------------------
# MB3.1
# --------------------------------------------------------------------------
bash "${PROJECT_ROOT}/scripts/run/run_mb31.sh" \
  $([[ "${DO_BUILD}" -eq 1 ]] && echo --build) \
  --device "${DEVICE}" \
  --trace-seconds "${TRACE_SECONDS}" \
  --output-dir "${OUTPUT_ROOT}/mb31"

# --------------------------------------------------------------------------
# MB3.2
# --------------------------------------------------------------------------
bash "${PROJECT_ROOT}/scripts/run/run_mb32.sh" \
  $([[ "${DO_BUILD}" -eq 1 ]] && echo --build) \
  --device "${DEVICE}" \
  --reg-footprint 64 \
  --shared-bytes 32768 \
  --block 256 \
  --grid 120 \
  --warmup 5 \
  --reps 20 \
  --output-dir "${OUTPUT_ROOT}/mb32"

# --------------------------------------------------------------------------
# MB3.4
# --------------------------------------------------------------------------
bash "${PROJECT_ROOT}/scripts/run/run_mb34.sh" \
  $([[ "${DO_BUILD}" -eq 1 ]] && echo --build) \
  --device "${DEVICE}" \
  --n 67108864 \
  --block 256 \
  --grid 120 \
  --warmup 5 \
  --reps 20 \
  --working-set 1048576 \
  --iters 1048576 \
  --seed 12345 \
  --output-dir "${OUTPUT_ROOT}/mb34"

# --------------------------------------------------------------------------
# MB8 / MB3 / MB10
# --------------------------------------------------------------------------
bash "${PROJECT_ROOT}/scripts/run/run_mb_core_extra.sh" \
  $([[ "${DO_BUILD}" -eq 1 ]] && echo --build) \
  --device "${DEVICE}" \
  --stride 1 \
  --n 16777216 \
  --fmas-per-load 64 \
  --block 256 \
  --grid 120 \
  --warmup 5 \
  --reps 20 \
  --output-dir "${OUTPUT_ROOT}/mb_core_extra"

# --------------------------------------------------------------------------
# cuBLASLt baseline
# --------------------------------------------------------------------------
bash "${PROJECT_ROOT}/scripts/run/run_mb33_baseline.sh" \
  $([[ "${DO_BUILD}" -eq 1 ]] && echo --build) \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --m "${M}" \
  --n "${N}" \
  --k "${K}" \
  --warmup 5 \
  --reps 20 \
  --output-dir "${OUTPUT_ROOT}/mb33_baseline"

echo "[campaign_mb31_to_mb35] Done."
echo "[campaign_mb31_to_mb35] Results stored under:"
echo "  ${OUTPUT_ROOT}"