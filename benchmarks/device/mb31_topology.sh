#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# MB3.1-C: GPU topology / interconnect capture
#
# Usage:
#   bash benchmarks/device/mb31_topology.sh [output_dir]
#
# Outputs:
#   topo_matrix.txt
#   nvlink_status.txt
#   gpu_list.txt
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

OUT_DIR="${1:-${RAW_RESULTS_DIR}/mb31_${RUN_TAG}}"
mkdir -p "${OUT_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[mb31_topology] ERROR: nvidia-smi not found." >&2
  exit 1
fi

echo "[mb31_topology] Output directory: ${OUT_DIR}"

echo "[mb31_topology] Capturing GPU list..."
nvidia-smi -L > "${OUT_DIR}/gpu_list.txt"

echo "[mb31_topology] Capturing topology matrix..."
nvidia-smi topo -m > "${OUT_DIR}/topo_matrix.txt"

echo "[mb31_topology] Capturing NVLink status..."
# Some systems may not support nvlink query cleanly; do not fail hard here
if nvidia-smi nvlink -s > "${OUT_DIR}/nvlink_status.txt" 2>"${OUT_DIR}/nvlink_status.err"; then
  :
else
  echo "[mb31_topology] WARNING: nvidia-smi nvlink -s failed; see nvlink_status.err"
fi

echo "[mb31_topology] Done."