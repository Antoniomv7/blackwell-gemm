#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0
OUTPUT_DIR=""
DTYPE="fp16"
M=4096
N=4096
K=4096

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *)
      echo "[run_mb35] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RAW_RESULTS_DIR}/mb35_${RUN_TAG}"
fi
mkdir -p "${OUTPUT_DIR}"

resolve_source_path() {
  local target="$1"
  local src
  src="$(awk -v key="${target}_SRC" '$1==key && $2==":=" { for (i=3;i<=NF;i++) printf "%s%s", $i, (i<NF?OFS:ORS) }' "${PROJECT_ROOT}/Makefile")"
  src="${src//\$\(PROJECT_ROOT\)/${PROJECT_ROOT}}"
  src="${src//\$\(DEVICE_DIR\)/${PROJECT_ROOT}/benchmarks/device}"
  src="${src//\$\(SM_DIR\)/${PROJECT_ROOT}/benchmarks/sm}"
  src="${src//\$\(MEMORY_DIR\)/${PROJECT_ROOT}/benchmarks/memory}"
  src="${src//\$\(TENSOR_DIR\)/${PROJECT_ROOT}/benchmarks/tensor}"
  src="${src//\$\(BENCH_DIR\)/${PROJECT_ROOT}/benchmarks}"
  printf '%s\n' "${src}"
}

ensure_binary() {
  local target="$1"
  local src_path bin_path
  bin_path="${BIN_DIR}/${target}"
  src_path="$(resolve_source_path "${target}")"

  if [[ -z "${src_path}" ]]; then
    echo "[run_mb35] ERROR: could not resolve source path for ${target}" >&2
    exit 1
  fi
  if [[ ! -f "${src_path}" ]]; then
    echo "[run_mb35] ERROR: source file for ${target} not found: ${src_path}" >&2
    exit 1
  fi

  if [[ "${DO_BUILD}" -eq 1 || ! -x "${bin_path}" || "${src_path}" -nt "${bin_path}" ]]; then
    echo "[run_mb35] Building ${target}..."
    bash "${PROJECT_ROOT}/scripts/run/build_one.sh" "${target}"
  fi

  if [[ ! -x "${bin_path}" ]]; then
    echo "[run_mb35] ERROR: binary not found or not executable after build: ${bin_path}" >&2
    exit 1
  fi
}

run_with_energy() {
  local target="$1"
  shift
  local out_prefix="$1"
  shift

  local trace_csv="${OUTPUT_DIR}/${out_prefix}_power_trace.csv"
  local energy_json="${OUTPUT_DIR}/${out_prefix}_energy.json"
  local trace_pid=""

  cleanup_trace() {
    if [[ -n "${trace_pid}" ]]; then
      kill "${trace_pid}" >/dev/null 2>&1 || true
      wait "${trace_pid}" 2>/dev/null || true
    fi
  }

  ensure_binary "${target}"

  echo "[run_mb35] Running ${target} with telemetry..."
  bash "${PROJECT_ROOT}/benchmarks/device/mb31_power_trace.sh" "${trace_csv}" &
  trace_pid=$!
  sleep 0.3

  "${BIN_DIR}/${target}" "$@"

  cleanup_trace

  python3 "${PROJECT_ROOT}/analysis/nvml_energy.py" \
    --trace-csv "${trace_csv}" \
    --out-json "${energy_json}"
}

run_with_energy \
  mb34_stream_triad \
  mb34_stream_triad \
  --device "${DEVICE}" \
  --n 67108864 \
  --block 256 \
  --grid 120 \
  --warmup 5 \
  --reps 20 \
  --csv "${OUTPUT_DIR}/mb34_stream_triad.csv"

run_with_energy \
  mb4_mb5_cublaslt_baseline \
  mb4_mb5_cublaslt_baseline \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --m "${M}" \
  --n "${N}" \
  --k "${K}" \
  --warmup 5 \
  --reps 20 \
  --csv "${OUTPUT_DIR}/mb4_mb5_cublaslt_baseline.csv"

echo "[run_mb35] Done."
echo "[run_mb35] Output dir: ${OUTPUT_DIR}"
