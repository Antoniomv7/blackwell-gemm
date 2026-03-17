#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

echo "[post_create] Validating environment..."
python -V
pip -V
nvcc --version || true

bw_env_summary || true
bash "${PROJECT_ROOT}/scripts/env/check_requirements.sh"

echo "[post_create] Environment ready."
