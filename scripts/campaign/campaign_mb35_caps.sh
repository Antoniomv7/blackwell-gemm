#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0
POWER_CAPS=""
OUTPUT_ROOT=""
DTYPE="fp16"
M=4096
N=4096
K=4096
ORIGINAL_POWER_LIMIT=""
RESTORE_POWER_LIMIT=0
POWER_CAP_CONTROL_AVAILABLE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --power-caps) POWER_CAPS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "[campaign_mb35_caps] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${POWER_CAPS}" ]]; then
  echo "[campaign_mb35_caps] ERROR: --power-caps is required." >&2
  exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${RESULTS_DIR}/mb35_caps_${RUN_TAG}"
fi
mkdir -p "${OUTPUT_ROOT}"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb34_stream_triad
  bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb4_mb5_cublaslt_baseline
fi

check_power_cap_control() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[campaign_mb35_caps] WARNING: nvidia-smi not found. Power-cap control is unavailable." >&2
    return 1
  fi

  local queried
  queried="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "${DEVICE}" 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ -z "${queried}" ]]; then
    echo "[campaign_mb35_caps] WARNING: Could not query current power limit on device ${DEVICE}." >&2
    return 1
  fi

  ORIGINAL_POWER_LIMIT="${queried}"
  RESTORE_POWER_LIMIT=1
  POWER_CAP_CONTROL_AVAILABLE=1
  return 0
}

restore_power_limit() {
  if [[ "${RESTORE_POWER_LIMIT}" -eq 1 ]]; then
    echo "[campaign_mb35_caps] Restoring power limit to ${ORIGINAL_POWER_LIMIT} W on device ${DEVICE}" >&2
    nvidia-smi -i "${DEVICE}" -pl "${ORIGINAL_POWER_LIMIT}" >/dev/null 2>&1 || true
  fi
}
trap restore_power_limit EXIT

set_power_limit() {
  local cap="$1"
  if [[ "${POWER_CAP_CONTROL_AVAILABLE}" -ne 1 ]]; then
    echo "[campaign_mb35_caps] ERROR: power-cap control is not available in this environment." >&2
    exit 1
  fi

  echo "[campaign_mb35_caps] Setting power limit to ${cap} W on device ${DEVICE}"
  if ! nvidia-smi -i "${DEVICE}" -pl "${cap}" >/dev/null 2>&1; then
    echo "[campaign_mb35_caps] ERROR: failed to set power limit to ${cap} W on device ${DEVICE}." >&2
    echo "[campaign_mb35_caps] This usually means insufficient permissions or cluster policy restrictions." >&2
    exit 1
  fi
}

check_power_cap_control
IFS=',' read -r -a CAPS <<< "${POWER_CAPS}"

for cap in "${CAPS[@]}"; do
  CAP_DIR="${OUTPUT_ROOT}/pl_${cap}W"
  mkdir -p "${CAP_DIR}"

  set_power_limit "${cap}"

  bash "${PROJECT_ROOT}/scripts/profile/profile_power_trace.sh" \
    --target mb34_stream_triad \
    --out-prefix "triad_pl_${cap}W" \
    -- \
    --device "${DEVICE}" \
    --n 67108864 \
    --block 256 \
    --grid 120 \
    --warmup 5 \
    --reps 20 \
    --csv "${CAP_DIR}/mb34_stream_triad.csv"

  bash "${PROJECT_ROOT}/scripts/profile/profile_power_trace.sh" \
    --target mb4_mb5_cublaslt_baseline \
    --out-prefix "gemm_pl_${cap}W" \
    -- \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --m "${M}" \
    --n "${N}" \
    --k "${K}" \
    --warmup 5 \
    --reps 20 \
    --csv "${CAP_DIR}/mb4_mb5_cublaslt_baseline.csv"
done

echo "[campaign_mb35_caps] Done."
echo "[campaign_mb35_caps] Output root: ${OUTPUT_ROOT}"
