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
ORIGINAL_POWER_LIMIT=""
RESTORE_POWER_LIMIT=0
POWER_CAP_CONTROL_AVAILABLE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --power-caps) POWER_CAPS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "[campaign_power_sweep] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${RESULTS_DIR}/power_sweep_${RUN_TAG}"
fi
mkdir -p "${OUTPUT_ROOT}"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb34_stream_triad
  bash "${PROJECT_ROOT}/scripts/run/build_one.sh" mb4_mb5_cublaslt_baseline
fi

check_power_cap_control() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[campaign_power_sweep] WARNING: nvidia-smi not found. Power-cap control is unavailable." >&2
    return 1
  fi

  local queried
  queried="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "${DEVICE}" 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ -z "${queried}" ]]; then
    echo "[campaign_power_sweep] WARNING: Could not query current power limit on device ${DEVICE}." >&2
    return 1
  fi

  ORIGINAL_POWER_LIMIT="${queried}"
  RESTORE_POWER_LIMIT=1
  POWER_CAP_CONTROL_AVAILABLE=1
  return 0
}

restore_power_limit() {
  if [[ "${RESTORE_POWER_LIMIT}" -eq 1 ]]; then
    echo "[campaign_power_sweep] Restoring power limit to ${ORIGINAL_POWER_LIMIT} W on device ${DEVICE}" >&2
    nvidia-smi -i "${DEVICE}" -pl "${ORIGINAL_POWER_LIMIT}" >/dev/null 2>&1 || true
  fi
}
trap restore_power_limit EXIT

set_power_limit() {
  local cap="$1"
  if [[ "${POWER_CAP_CONTROL_AVAILABLE}" -ne 1 ]]; then
    echo "[campaign_power_sweep] ERROR: power-cap control is not available in this environment." >&2
    exit 1
  fi

  echo "[campaign_power_sweep] Setting power limit to ${cap} W on device ${DEVICE}"
  if ! nvidia-smi -i "${DEVICE}" -pl "${cap}" >/dev/null 2>&1; then
    echo "[campaign_power_sweep] ERROR: failed to set power limit to ${cap} W on device ${DEVICE}." >&2
    echo "[campaign_power_sweep] This usually means insufficient permissions or cluster policy restrictions." >&2
    exit 1
  fi
}

run_pair() {
  local tag="$1"
  local out_dir="${OUTPUT_ROOT}/${tag}"
  mkdir -p "${out_dir}"

  echo "[campaign_power_sweep] Running triad with trace: ${tag}"
  bash "${PROJECT_ROOT}/scripts/run/run_with_trace.sh" \
    --target mb34_stream_triad \
    -- \
    --device "${DEVICE}" \
    --n 67108864 \
    --block 256 \
    --grid 120 \
    --warmup 5 \
    --reps 20 \
    --csv "${out_dir}/mb34_stream_triad.csv"

  echo "[campaign_power_sweep] Running cuBLASLt baseline with trace: ${tag}"
  bash "${PROJECT_ROOT}/scripts/run/run_with_trace.sh" \
    --target mb4_mb5_cublaslt_baseline \
    -- \
    --device "${DEVICE}" \
    --dtype fp16 \
    --m 4096 \
    --n 4096 \
    --k 4096 \
    --warmup 5 \
    --reps 20 \
    --csv "${out_dir}/mb4_mb5_cublaslt_baseline.csv"
}

if [[ -z "${POWER_CAPS}" ]]; then
  echo "[campaign_power_sweep] No power caps provided. Running only default-power campaign."
  run_pair "default_power"
else
  check_power_cap_control
  IFS=',' read -r -a CAPS <<< "${POWER_CAPS}"
  for cap in "${CAPS[@]}"; do
    set_power_limit "${cap}"
    run_pair "pl_${cap}W"
  done
fi

echo "[campaign_power_sweep] Done."
echo "[campaign_power_sweep] Output root:"
echo "  ${OUTPUT_ROOT}"
