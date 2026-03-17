#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Build one target from the Makefile.
#
# Usage:
#   bash scripts/run/build_one.sh mb31_inventory
#   bash scripts/run/build_one.sh mb34_stream_triad
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

if [[ $# -lt 1 ]]; then
  echo "[build_one] Usage: $0 <target>" >&2
  exit 1
fi

TARGET="$1"

cd "${PROJECT_ROOT}"

echo "[build_one] Project root: ${PROJECT_ROOT}"
echo "[build_one] Building target: ${TARGET}"
echo "[build_one] CUDA_ARCH=${CUDA_ARCH}"

make "${TARGET}" CUDA_ARCH="${CUDA_ARCH}"

echo "[build_one] Done: ${TARGET}"