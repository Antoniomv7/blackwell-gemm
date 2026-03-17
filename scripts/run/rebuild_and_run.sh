#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Clean build artifacts, rebuild everything available, and run the main campaign.
#
# Usage:
#   bash scripts/run/rebuild_and_run.sh
#   bash scripts/run/rebuild_and_run.sh --device 0
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DEVICE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    *)
      echo "[rebuild_and_run] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "[rebuild_and_run] Cleaning build artifacts..."
bash "${PROJECT_ROOT}/scripts/run/clean_results.sh" --build

echo "[rebuild_and_run] Rebuilding all available targets..."
bash "${PROJECT_ROOT}/scripts/run/build_all.sh"

echo "[rebuild_and_run] Running main campaign..."
bash "${PROJECT_ROOT}/scripts/campaign/campaign_mb31_to_mb35.sh" \
  --device "${DEVICE}"

echo "[rebuild_and_run] Done."