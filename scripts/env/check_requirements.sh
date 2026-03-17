#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

MISSING_REQUIRED=0
MISSING_OPTIONAL=0

check_required() {
  local cmd="$1"
  local label="$2"
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ok] ${label}: $(command -v "${cmd}")"
  else
    echo "[missing] ${label}: ${cmd}"
    MISSING_REQUIRED=1
  fi
}

check_optional() {
  local cmd="$1"
  local label="$2"
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ok] optional ${label}: $(command -v "${cmd}")"
  else
    echo "[missing] optional ${label}: ${cmd}"
    MISSING_OPTIONAL=1
  fi
}

echo "[check] Project root: ${PROJECT_ROOT}"
echo "[check] Run tag: ${RUN_TAG}"
echo

echo "[check] Required tools"
check_required bash "Bash"
check_required awk "awk"
check_required sed "sed"
check_required grep "grep"
check_required python3 "Python 3"
check_required nvidia-smi "NVIDIA SMI"
check_required nvcc "CUDA compiler"
check_required make "GNU Make"
check_required gcc "GCC"

echo
printf '[check] Optional but recommended tools\n'
check_optional ncu "Nsight Compute CLI"
check_optional nsys "Nsight Systems CLI"
check_optional git "Git"

echo
printf '[check] Python modules (best effort)\n'
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
mods = ["numpy", "pandas", "matplotlib", "yaml"]
optional = ["pynvml", "pycuda"]
for m in mods:
    try:
        __import__(m)
        print(f"[ok] python module: {m}")
    except Exception:
        print(f"[missing] python module: {m}")
for m in optional:
    try:
        __import__(m)
        print(f"[ok] optional python module: {m}")
    except Exception:
        print(f"[missing] optional python module: {m}")
PY
fi

echo
printf '[check] GPU visibility\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

echo
printf '[check] CUDA compiler version\n'
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
fi

echo
printf '[check] Environment summary\n'
bw_env_summary || true

echo
if [[ "${MISSING_REQUIRED}" -ne 0 ]]; then
  echo "[check] One or more required tools are missing." >&2
  exit 1
fi

if [[ "${MISSING_OPTIONAL}" -ne 0 ]]; then
  echo "[check] Optional tooling is incomplete. Core benchmark flow may still work, but profiling helpers that use ncu/nsys will not." >&2
fi

echo "[check] Required tools satisfied."
