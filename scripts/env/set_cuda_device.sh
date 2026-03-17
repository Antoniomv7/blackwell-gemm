#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Set CUDA_VISIBLE_DEVICES explicitly.
#
# Modes:
#   --gpu <index>     Use the specified GPU index
#   (no args)         Auto-detect a candidate GPU with detect_gpu.sh
#   --print-only      Print the export line instead of exporting in current shell
#
# Typical usage:
#   source scripts/env/set_cuda_device.sh
#   source scripts/env/set_cuda_device.sh --gpu 3
#
# Important:
#   To modify the current shell environment, this script should be sourced.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

GPU_INDEX=""
PRINT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      if [[ $# -lt 2 ]]; then
        echo "[set_cuda_device] ERROR: --gpu requires an argument." >&2
        exit 1
      fi
      GPU_INDEX="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    *)
      echo "[set_cuda_device] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${GPU_INDEX}" ]]; then
  GPU_INDEX="$("${SCRIPT_DIR}/detect_gpu.sh")"
fi

if [[ -z "${GPU_INDEX}" ]]; then
  echo "[set_cuda_device] ERROR: no GPU index available." >&2
  exit 1
fi

# Validate the selected index exists
if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits | grep -qx "${GPU_INDEX}"; then
  echo "[set_cuda_device] ERROR: GPU index ${GPU_INDEX} is not visible via nvidia-smi." >&2
  exit 1
fi

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  echo "export CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

echo "[set_cuda_device] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[set_cuda_device] Visible GPU summary:"
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,pstate \
  --format=csv