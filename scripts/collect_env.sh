#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/env_$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "=== TIMESTAMP ==="
  date -Is
  echo

  echo "=== HOST ==="
  hostname
  uname -a
  echo

  echo "=== GIT ==="
  git rev-parse HEAD 2>/dev/null || echo "no git"
  git status --porcelain 2>/dev/null || true
  echo

  echo "=== CONTAINER ==="
  cat /etc/os-release || true
  echo "PATH=$PATH"
  echo

  echo "=== PYTHON ==="
  which python || true
  python -V || true
  pip -V || true
  echo

  echo "=== NVIDIA / GPU ==="
  command -v nvidia-smi && nvidia-smi -L || echo "nvidia-smi not found"
  command -v nvidia-smi && nvidia-smi || true
  echo

  echo "=== CUDA ==="
  command -v nvcc && nvcc --version || echo "nvcc not found"
  echo

} | tee "$OUT"
echo "[collect_env] wrote $OUT"