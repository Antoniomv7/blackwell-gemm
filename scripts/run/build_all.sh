#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Build all currently existing targets in the project.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

cd "${PROJECT_ROOT}"

echo "[build_all] Project root: ${PROJECT_ROOT}"
echo "[build_all] CUDA_ARCH=${CUDA_ARCH}"
echo "[build_all] Building all existing targets..."

make all CUDA_ARCH="${CUDA_ARCH}"

echo "[build_all] Done."