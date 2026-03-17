#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Nsight Compute profiling for MB3.4 selected runs.
#
# Usage:
#   bash scripts/profile/profile_mb34.sh triad
#   bash scripts/profile/profile_mb34.sh ptr
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

if [[ $# -lt 1 ]]; then
  echo "[profile_mb34] Usage: $0 {triad|ptr}" >&2
  exit 1
fi

MODE="$1"
OUT_DIR="${NCU_RESULTS_DIR}/mb34_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "[profile_mb34] ERROR: ncu not found." >&2
  exit 1
fi

case "${MODE}" in
  triad)
    bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb34_stream_triad
    echo "[profile_mb34] Profiling stream triad..."
    ncu \
      --set full \
      --target-processes all \
      --export "${OUT_DIR}/mb34_stream_triad" \
      "${BIN_DIR}/mb34_stream_triad" \
      --n 67108864 \
      --block 256 \
      --warmup 3 \
      --reps 5
    ;;
  ptr)
    bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb34_ptr_chase
    echo "[profile_mb34] Profiling pointer chasing..."
    ncu \
      --set full \
      --target-processes all \
      --export "${OUT_DIR}/mb34_ptr_chase" \
      "${BIN_DIR}/mb34_ptr_chase" \
      --working-set 1048576 \
      --iters 1048576
    ;;
  *)
    echo "[profile_mb34] Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

echo "[profile_mb34] Done. Output dir: ${OUT_DIR}"