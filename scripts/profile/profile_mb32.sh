#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Nsight Compute profiling for MB3.2 selected runs.
#
# Usage:
#   bash scripts/profile/profile_mb32.sh residency
#   bash scripts/profile/profile_mb32.sh fp32
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

if [[ $# -lt 1 ]]; then
  echo "[profile_mb32] Usage: $0 {residency|fp32}" >&2
  exit 1
fi

MODE="$1"
OUT_DIR="${NCU_RESULTS_DIR}/mb32_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "[profile_mb32] ERROR: ncu not found." >&2
  exit 1
fi

case "${MODE}" in
  residency)
    bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb32_residency
    echo "[profile_mb32] Profiling residency probe..."
    ncu \
      --set full \
      --target-processes all \
      --export "${OUT_DIR}/mb32_residency" \
      "${BIN_DIR}/mb32_residency" \
      --reg-footprint 64 \
      --shared-bytes 32768 \
      --block 256 \
      --grid 120 \
      --warmup 3 \
      --reps 5
    ;;
  fp32)
    bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb32_fp32_probe
    echo "[profile_mb32] Profiling FP32 probe..."
    ncu \
      --set full \
      --target-processes all \
      --export "${OUT_DIR}/mb32_fp32_probe" \
      "${BIN_DIR}/mb32_fp32_probe" \
      --block 256 \
      --grid 120 \
      --warmup 3 \
      --reps 5
    ;;
  *)
    echo "[profile_mb32] Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

echo "[profile_mb32] Done. Output dir: ${OUT_DIR}"