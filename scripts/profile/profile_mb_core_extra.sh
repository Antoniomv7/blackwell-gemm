#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Nsight Compute profiling for MB8 / MB3 / MB10 selected runs.
#
# Usage:
#   bash scripts/profile/profile_mb_core_extra.sh mb8
#   bash scripts/profile/profile_mb_core_extra.sh mb3
#   bash scripts/profile/profile_mb_core_extra.sh mb10
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

if [[ $# -lt 1 ]]; then
  echo "[profile_mb_core_extra] Usage: $0 {mb8|mb3|mb10}" >&2
  exit 1
fi

MODE="$1"
OUT_DIR="${NCU_RESULTS_DIR}/mb_core_extra_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "[profile_mb_core_extra] ERROR: ncu not found." >&2
  exit 1
fi

case "${MODE}" in
  mb8)
    bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb8_shared_stride
    echo "[profile_mb_core_extra] Profiling MB8..."
    ncu \
      --set full \
      --target-processes all \
      --export "${OUT_DIR}/mb8_shared_stride" \
      "${BIN_DIR}/mb8_shared_stride" \
      --stride 1 \
      --block 256 \
      --grid 120 \
      --warmup 3 \
      --reps 5
    ;;
  mb3)
    bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb3_mem_compute_mix
    echo "[profile_mb_core_extra] Profiling MB3..."
    ncu \
      --set full \
      --target-processes all \
      --export "${OUT_DIR}/mb3_mem_compute_mix" \
      "${BIN_DIR}/mb3_mem_compute_mix" \
      --n 16777216 \
      --block 256 \
      --grid 120 \
      --fmas-per-load 64 \
      --warmup 3 \
      --reps 5
    ;;
  mb10)
    bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb10_ai_control
    echo "[profile_mb_core_extra] Profiling MB10..."
    ncu \
      --set full \
      --target-processes all \
      --export "${OUT_DIR}/mb10_ai_control" \
      "${BIN_DIR}/mb10_ai_control" \
      --n 16777216 \
      --block 256 \
      --grid 120 \
      --fmas-per-load 64 \
      --warmup 3 \
      --reps 5
    ;;
  *)
    echo "[profile_mb_core_extra] Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

echo "[profile_mb_core_extra] Done. Output dir: ${OUT_DIR}"