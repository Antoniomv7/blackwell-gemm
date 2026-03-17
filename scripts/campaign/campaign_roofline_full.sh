#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Full roofline campaign:
#   1) run MB3.4-A stream triad
#   2) run MB3 and MB10
#   3) build roofline points and plot
#
# Usage:
#   bash scripts/campaign/campaign_roofline_full.sh --build --peak-compute-gflops 80000
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

DO_BUILD=0
DEVICE=0
PEAK_COMPUTE_GFLOPS=""
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --peak-compute-gflops) PEAK_COMPUTE_GFLOPS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "[campaign_roofline_full] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PEAK_COMPUTE_GFLOPS}" ]]; then
  echo "[campaign_roofline_full] ERROR: --peak-compute-gflops is required." >&2
  exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${RESULTS_DIR}/roofline_campaign_${RUN_TAG}"
fi

RAW_DIR="${OUTPUT_ROOT}/raw"
PROC_DIR="${OUTPUT_ROOT}/processed"
mkdir -p "${RAW_DIR}" "${PROC_DIR}"

# --------------------------------------------------------------------------
# Run triad
# --------------------------------------------------------------------------
bash "${PROJECT_ROOT}/scripts/run/run_mb34.sh" \
  $([[ "${DO_BUILD}" -eq 1 ]] && echo --build) \
  --device "${DEVICE}" \
  --n 67108864 \
  --block 256 \
  --grid 120 \
  --warmup 5 \
  --reps 20 \
  --working-set 1048576 \
  --iters 1048576 \
  --output-dir "${RAW_DIR}/mb34"

# --------------------------------------------------------------------------
# Run MB3 + MB10
# --------------------------------------------------------------------------
bash "${PROJECT_ROOT}/scripts/run/run_mb_core_extra.sh" \
  $([[ "${DO_BUILD}" -eq 1 ]] && echo --build) \
  --device "${DEVICE}" \
  --stride 1 \
  --n 16777216 \
  --fmas-per-load 64 \
  --block 256 \
  --grid 120 \
  --warmup 5 \
  --reps 20 \
  --output-dir "${RAW_DIR}/mb_core_extra"

TRIAD_CSV="${RAW_DIR}/mb34/mb34_stream_triad.csv"
AI_CONTROL_CSV="${RAW_DIR}/mb_core_extra/mb10_ai_control.csv"
MEM_COMPUTE_CSV="${RAW_DIR}/mb_core_extra/mb3_mem_compute_mix.csv"

if [[ ! -f "${TRIAD_CSV}" ]]; then
  echo "[campaign_roofline_full] ERROR: triad CSV not found: ${TRIAD_CSV}" >&2
  exit 1
fi

PEAK_BW_GBS="$(python3 - <<PY
import csv
with open("${TRIAD_CSV}", "r", encoding="utf-8", newline="") as f:
    row = next(csv.DictReader(f))
print(row["bw_best_gbs"])
PY
)"

echo "[campaign_roofline_full] Derived peak BW from triad: ${PEAK_BW_GBS} GB/s"

bash "${PROJECT_ROOT}/scripts/run/run_roofline.sh" \
  --triad-csv "${TRIAD_CSV}" \
  --ai-control-csv "${AI_CONTROL_CSV}" \
  --mem-compute-csv "${MEM_COMPUTE_CSV}" \
  --peak-compute-gflops "${PEAK_COMPUTE_GFLOPS}" \
  --peak-bw-gbs "${PEAK_BW_GBS}" \
  --output-dir "${PROC_DIR}"

echo "[campaign_roofline_full] Done."
echo "[campaign_roofline_full] Output root:"
echo "  ${OUTPUT_ROOT}"