#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run one benchmark binary in a unified way.
#
# Usage:
#   bash scripts/run/run_one.sh --target mb31_inventory
#   bash scripts/run/run_one.sh --target mb34_stream_triad -- --n 100000000
#   bash scripts/run/run_one.sh --build --target mb34_stream_triad -- --n 100000000
#
# Notes:
#   - arguments after "--" are forwarded to the benchmark binary
#   - use source scripts/env/set_cuda_device.sh first if needed
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

TARGET=""
DO_BUILD=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      if [[ $# -lt 2 ]]; then
        echo "[run_one] ERROR: --target requires a value." >&2
        exit 1
      fi
      TARGET="$2"
      shift 2
      ;;
    --build)
      DO_BUILD=1
      shift
      ;;
    --)
      shift
      FORWARD_ARGS=("$@")
      break
      ;;
    *)
      echo "[run_one] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TARGET}" ]]; then
  echo "[run_one] Usage: $0 --target <name> [--build] [-- <benchmark args>]" >&2
  exit 1
fi

BIN_PATH="${BIN_DIR}/${TARGET}"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/build_one.sh" "${TARGET}"
fi

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "[run_one] ERROR: binary not found or not executable: ${BIN_PATH}" >&2
  echo "[run_one] Hint: use --build or run build_one.sh first." >&2
  exit 1
fi

echo "[run_one] Target: ${TARGET}"
echo "[run_one] Binary: ${BIN_PATH}"
echo "[run_one] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[run_one] RUN_TAG=${RUN_TAG}"

if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
  echo "[run_one] Forwarded args: ${FORWARD_ARGS[*]}"
fi

"${BIN_PATH}" "${FORWARD_ARGS[@]}"