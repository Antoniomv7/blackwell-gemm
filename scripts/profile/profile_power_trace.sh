#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

TARGET=""
OUT_PREFIX=""
NO_BUILD=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --out-prefix)
      OUT_PREFIX="$2"
      shift 2
      ;;
    --no-build)
      NO_BUILD=1
      shift
      ;;
    --)
      shift
      FORWARD_ARGS=("$@")
      break
      ;;
    *)
      echo "[profile_power_trace] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TARGET}" || -z "${OUT_PREFIX}" ]]; then
  echo "[profile_power_trace] Usage: $0 --target <bin> --out-prefix <name> [--no-build] -- <args>" >&2
  exit 1
fi

if ! grep -qE "^[[:space:]]*${TARGET}_SRC[[:space:]]*:=" "${PROJECT_ROOT}/Makefile"; then
  echo "[profile_power_trace] ERROR: target '${TARGET}' is not declared in the Makefile." >&2
  exit 1
fi

SRC_PATH="$(awk -v target="${TARGET}_SRC" '$1 == target && $2 == ":=" { for (i=3;i<=NF;i++) printf "%s%s", $i, (i<NF?OFS:ORS) }' "${PROJECT_ROOT}/Makefile")"
if [[ -z "${SRC_PATH}" ]]; then
  echo "[profile_power_trace] ERROR: could not resolve source path for target '${TARGET}'." >&2
  exit 1
fi

# Expand common Makefile vars into absolute paths for existence checks.
SRC_PATH="${SRC_PATH//\$\(PROJECT_ROOT\)/${PROJECT_ROOT}}"
SRC_PATH="${SRC_PATH//\$\(DEVICE_DIR\)/${PROJECT_ROOT}/benchmarks/device}"
SRC_PATH="${SRC_PATH//\$\(SM_DIR\)/${PROJECT_ROOT}/benchmarks/sm}"
SRC_PATH="${SRC_PATH//\$\(MEMORY_DIR\)/${PROJECT_ROOT}/benchmarks/memory}"
SRC_PATH="${SRC_PATH//\$\(TENSOR_DIR\)/${PROJECT_ROOT}/benchmarks/tensor}"
SRC_PATH="${SRC_PATH//\$\(BENCH_DIR\)/${PROJECT_ROOT}/benchmarks}"

if [[ ! -f "${SRC_PATH}" ]]; then
  echo "[profile_power_trace] ERROR: source file for '${TARGET}' not found: ${SRC_PATH}" >&2
  exit 1
fi

BIN_PATH="${BIN_DIR}/${TARGET}"
if [[ ! -x "${BIN_PATH}" ]]; then
  if [[ "${NO_BUILD}" -eq 1 ]]; then
    echo "[profile_power_trace] ERROR: binary not found and --no-build was given: ${BIN_PATH}" >&2
    exit 1
  fi
  echo "[profile_power_trace] Binary missing; building ${TARGET}..."
  bash "${PROJECT_ROOT}/scripts/run/build_one.sh" "${TARGET}"
elif [[ "${SRC_PATH}" -nt "${BIN_PATH}" ]]; then
  if [[ "${NO_BUILD}" -eq 1 ]]; then
    echo "[profile_power_trace] ERROR: source is newer than binary and --no-build was given." >&2
    echo "  source: ${SRC_PATH}" >&2
    echo "  binary: ${BIN_PATH}" >&2
    exit 1
  fi
  echo "[profile_power_trace] Source newer than binary; rebuilding ${TARGET}..."
  bash "${PROJECT_ROOT}/scripts/run/build_one.sh" "${TARGET}"
fi

OUT_DIR="${POWER_RESULTS_DIR}/trace_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

TRACE_CSV="${OUT_DIR}/${OUT_PREFIX}_power_trace.csv"
ENERGY_JSON="${OUT_DIR}/${OUT_PREFIX}_energy.json"

cleanup() {
  if [[ -n "${TRACE_PID:-}" ]]; then
    kill "${TRACE_PID}" >/dev/null 2>&1 || true
    wait "${TRACE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[profile_power_trace] Starting trace -> ${TRACE_CSV}"
bash "${PROJECT_ROOT}/benchmarks/device/mb31_power_trace.sh" "${TRACE_CSV}" &
TRACE_PID=$!

sleep 0.3

echo "[profile_power_trace] Running target: ${TARGET}"
"${BIN_PATH}" "${FORWARD_ARGS[@]}"

cleanup
unset TRACE_PID

python3 "${PROJECT_ROOT}/analysis/nvml_energy.py" \
  --trace-csv "${TRACE_CSV}" \
  --out-json "${ENERGY_JSON}"

echo "[profile_power_trace] Done."
echo "  ${TRACE_CSV}"
echo "  ${ENERGY_JSON}"
