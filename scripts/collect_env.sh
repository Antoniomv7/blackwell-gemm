#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_OUT="${PROJECT_ROOT}/results/system/env_$(date +%Y%m%d_%H%M%S).txt"
OUT="${1:-${DEFAULT_OUT}}"
mkdir -p "$(dirname "$OUT")"

{
  echo "=== TIMESTAMP ==="
  date -Is
  echo

  echo "=== PROJECT ROOT ==="
  echo "${PROJECT_ROOT}"
  echo

  echo "=== HOST ==="
  hostname
  uname -a
  echo

  echo "=== GIT ==="
  git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo "no git"
  git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null || true
  echo

  echo "=== CONTAINER / OS ==="
  cat /etc/os-release 2>/dev/null || true
  echo "PATH=$PATH"
  echo

  echo "=== PYTHON ==="
  command -v python || true
  python -V || true
  pip -V || true
  echo

  echo "=== NVIDIA / GPU ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi
  echo

  echo "=== CUDA ==="
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || true
  else
    echo "nvcc not found"
  fi
  echo
} | tee "$OUT"

echo "[collect_env] wrote $OUT"
