#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_INDEX="${1:-}"
IMAGE_NAME="${IMAGE_NAME:-blackwell-gemm:dev}"

if [[ -z "${GPU_INDEX}" ]]; then
  echo "Usage: $0 <physical_gpu_index>" >&2
  echo "Example: $0 6" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[run_blackwell_container] ERROR: docker not found." >&2
  exit 1
fi

echo "[run_blackwell_container] Project root: ${PROJECT_ROOT}"
echo "[run_blackwell_container] Building image: ${IMAGE_NAME}"
echo "[run_blackwell_container] Using physical GPU index: ${GPU_INDEX}"

docker build \
  -t "${IMAGE_NAME}" \
  -f "${PROJECT_ROOT}/docker/Dockerfile" \
  "${PROJECT_ROOT}"

echo "[run_blackwell_container] Launching container..."

docker run --rm -it \
  --gpus "device=${GPU_INDEX}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e BLACKWELL_GPU_INDEX="${GPU_INDEX}" \
  -v "${PROJECT_ROOT}:/workspace/blackwell-gemm" \
  -w /workspace/blackwell-gemm \
  "${IMAGE_NAME}" \
  bash