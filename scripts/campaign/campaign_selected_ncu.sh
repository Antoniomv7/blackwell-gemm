#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Selected Nsight Compute campaign for representative benchmarks.
# The output root is enforced by temporarily overriding NCU_RESULTS_DIR for the
# child profiling scripts.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

OUTPUT_ROOT="${NCU_RESULTS_DIR}/selected_${RUN_TAG}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "[campaign_selected_ncu] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

echo "[campaign_selected_ncu] Output root: ${OUTPUT_ROOT}"
run_profile() {
  local subdir="$1"
  shift
  mkdir -p "${OUTPUT_ROOT}/${subdir}"
  (
    export NCU_RESULTS_DIR="${OUTPUT_ROOT}/${subdir}"
    bash "$@"
  )
}

run_profile mb32_residency "${PROJECT_ROOT}/scripts/profile/profile_mb32.sh" residency
run_profile mb32_fp32 "${PROJECT_ROOT}/scripts/profile/profile_mb32.sh" fp32
run_profile mb34_triad "${PROJECT_ROOT}/scripts/profile/profile_mb34.sh" triad
run_profile mb34_ptr "${PROJECT_ROOT}/scripts/profile/profile_mb34.sh" ptr
run_profile mb8 "${PROJECT_ROOT}/scripts/profile/profile_mb_core_extra.sh" mb8
run_profile mb3 "${PROJECT_ROOT}/scripts/profile/profile_mb_core_extra.sh" mb3
run_profile mb10 "${PROJECT_ROOT}/scripts/profile/profile_mb_core_extra.sh" mb10
run_profile mb33_baseline "${PROJECT_ROOT}/scripts/profile/profile_mb33_baseline.sh"

echo "[campaign_selected_ncu] Done."
