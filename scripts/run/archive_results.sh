#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Archive a results directory into a .tar.gz file.
#
# Usage:
#   bash scripts/run/archive_results.sh --input results/raw/campaign_x
#   bash scripts/run/archive_results.sh --input results --output archive/results_x.tar.gz
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

INPUT=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *)
      echo "[archive_results] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT}" ]]; then
  echo "[archive_results] ERROR: --input is required." >&2
  exit 1
fi

if [[ ! -e "${INPUT}" ]]; then
  echo "[archive_results] ERROR: input path does not exist: ${INPUT}" >&2
  exit 1
fi

if [[ -z "${OUTPUT}" ]]; then
  mkdir -p "${PROJECT_ROOT}/archive"
  base="$(basename "${INPUT}")"
  OUTPUT="${PROJECT_ROOT}/archive/${base}_${RUN_TAG}.tar.gz"
fi

mkdir -p "$(dirname "${OUTPUT}")"

echo "[archive_results] Creating archive:"
echo "  input : ${INPUT}"
echo "  output: ${OUTPUT}"

tar -czf "${OUTPUT}" -C "$(dirname "${INPUT}")" "$(basename "${INPUT}")"

echo "[archive_results] Done."