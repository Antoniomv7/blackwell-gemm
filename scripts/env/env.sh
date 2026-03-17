#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Global environment for the blackwell-gemm project.
# This file is meant to be sourced, not executed directly.
# -----------------------------------------------------------------------------

# Resolve project root from this file location:
# scripts/env/env.sh -> project root = ../..
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Standard directories
export PROJECT_ROOT
export BENCHMARKS_DIR="${PROJECT_ROOT}/benchmarks"
export ANALYSIS_DIR="${PROJECT_ROOT}/analysis"
export SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
export RESULTS_DIR="${PROJECT_ROOT}/results"
export RAW_RESULTS_DIR="${RESULTS_DIR}/raw"
export POWER_RESULTS_DIR="${RESULTS_DIR}/power"
export NCU_RESULTS_DIR="${RESULTS_DIR}/ncu"
export PLOTS_DIR="${RESULTS_DIR}/plots"
export TABLES_DIR="${RESULTS_DIR}/tables"
export BIN_DIR="${PROJECT_ROOT}/bin"
export BUILD_DIR="${PROJECT_ROOT}/build"
export LOG_DIR="${PROJECT_ROOT}/logs"

# Create standard directories if they do not exist yet
mkdir -p \
  "${BENCHMARKS_DIR}" \
  "${ANALYSIS_DIR}" \
  "${SCRIPTS_DIR}" \
  "${RAW_RESULTS_DIR}" \
  "${POWER_RESULTS_DIR}" \
  "${NCU_RESULTS_DIR}" \
  "${PLOTS_DIR}" \
  "${TABLES_DIR}" \
  "${BIN_DIR}" \
  "${BUILD_DIR}" \
  "${LOG_DIR}"

# Default run tag for reproducibility
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Default benchmark timing policy
export DEFAULT_WARMUP="${DEFAULT_WARMUP:-5}"
export DEFAULT_REPS="${DEFAULT_REPS:-20}"

# Default kernel launch hints
export DEFAULT_BLOCK_SIZE="${DEFAULT_BLOCK_SIZE:-256}"
export DEFAULT_GRID_MULTIPLIER="${DEFAULT_GRID_MULTIPLIER:-20}"

# Logging / telemetry defaults
export POWER_SAMPLE_MS="${POWER_SAMPLE_MS:-100}"

# Preferred architecture flag (adjust later if needed)
export CUDA_ARCH="${CUDA_ARCH:-sm_103}"

# User-controlled GPU selection:
# - if already set by the user, keep it
# - otherwise leave unset until set_cuda_device.sh decides
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# Default compiler flags
export NVCC_COMMON_FLAGS="${NVCC_COMMON_FLAGS:--O3 -std=c++17 -lineinfo}"
export NVCC_ARCH_FLAGS="${NVCC_ARCH_FLAGS:--arch=${CUDA_ARCH}}"

# Nsight Compute defaults
export NCU_DEFAULT_SET="${NCU_DEFAULT_SET:-full}"

# Helper: print environment summary
bw_env_summary() {
  cat <<EOF
[env]
PROJECT_ROOT=${PROJECT_ROOT}
BIN_DIR=${BIN_DIR}
RESULTS_DIR=${RESULTS_DIR}
RUN_TAG=${RUN_TAG}
CUDA_ARCH=${CUDA_ARCH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}
DEFAULT_WARMUP=${DEFAULT_WARMUP}
DEFAULT_REPS=${DEFAULT_REPS}
POWER_SAMPLE_MS=${POWER_SAMPLE_MS}
EOF
}

# Helper: ensure we are running from project-aware environment
bw_require_project_root() {
  if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "[env] ERROR: PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
    return 1
  fi
}