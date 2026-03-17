#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Clean generated results and/or build artifacts.
#
# Usage:
#   bash scripts/run/clean_results.sh --results
#   bash scripts/run/clean_results.sh --build
#   bash scripts/run/clean_results.sh --all
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

CLEAN_RESULTS=0
CLEAN_BUILD=0
CLEAN_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) CLEAN_RESULTS=1; shift ;;
    --build) CLEAN_BUILD=1; shift ;;
    --all) CLEAN_ALL=1; shift ;;
    *)
      echo "[clean_results] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${CLEAN_ALL}" -eq 1 ]]; then
  CLEAN_RESULTS=1
  CLEAN_BUILD=1
fi

if [[ "${CLEAN_RESULTS}" -eq 0 && "${CLEAN_BUILD}" -eq 0 ]]; then
  echo "[clean_results] Nothing selected. Use --results, --build or --all." >&2
  exit 1
fi

if [[ "${CLEAN_RESULTS}" -eq 1 ]]; then
  echo "[clean_results] Removing generated results/logs..."
  rm -rf "${RAW_RESULTS_DIR}"/*
  rm -rf "${POWER_RESULTS_DIR}"/*
  rm -rf "${NCU_RESULTS_DIR}"/*
  rm -rf "${PLOTS_DIR}"/*
  rm -rf "${TABLES_DIR}"/*
  rm -rf "${LOG_DIR}"/*
fi

if [[ "${CLEAN_BUILD}" -eq 1 ]]; then
  echo "[clean_results] Removing build/bin artifacts..."
  rm -rf "${BUILD_DIR}"/*
  rm -rf "${BIN_DIR}"/*
fi

echo "[clean_results] Done."