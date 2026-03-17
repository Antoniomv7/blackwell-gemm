#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Build roofline points and plot from previously generated benchmark CSVs.
#
# Example:
#   bash scripts/run/run_roofline.sh \
#     --triad-csv results/raw/mb34_x/mb34_stream_triad.csv \
#     --ai-control-csv results/raw/mb_core_extra_x/mb10_ai_control.csv \
#     --mem-compute-csv results/raw/mb_core_extra_x/mb3_mem_compute_mix.csv \
#     --peak-compute-gflops 80000 \
#     --peak-bw-gbs 3500
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/env/env.sh"

TRIAD_CSV=""
AI_CONTROL_CSV=""
MEM_COMPUTE_CSV=""
PEAK_COMPUTE_GFLOPS=""
PEAK_BW_GBS=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --triad-csv) TRIAD_CSV="$2"; shift 2 ;;
    --ai-control-csv) AI_CONTROL_CSV="$2"; shift 2 ;;
    --mem-compute-csv) MEM_COMPUTE_CSV="$2"; shift 2 ;;
    --peak-compute-gflops) PEAK_COMPUTE_GFLOPS="$2"; shift 2 ;;
    --peak-bw-gbs) PEAK_BW_GBS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *)
      echo "[run_roofline] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PEAK_COMPUTE_GFLOPS}" || -z "${PEAK_BW_GBS}" ]]; then
  echo "[run_roofline] ERROR: --peak-compute-gflops and --peak-bw-gbs are required." >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RESULTS_DIR}/processed/roofline_${RUN_TAG}"
fi

mkdir -p "${OUTPUT_DIR}"

POINTS_CSV="${OUTPUT_DIR}/roofline_points.csv"
RIDGE_JSON="${OUTPUT_DIR}/ridge_point.json"
PLOT_PNG="${OUTPUT_DIR}/roofline.png"

BUILD_CMD=(
  python3 "${PROJECT_ROOT}/analysis/build_roofline_points.py"
  --peak-compute-gflops "${PEAK_COMPUTE_GFLOPS}"
  --peak-bw-gbs "${PEAK_BW_GBS}"
  --points-out "${POINTS_CSV}"
  --ridge-out "${RIDGE_JSON}"
)

if [[ -n "${TRIAD_CSV}" ]]; then
  BUILD_CMD+=(--triad-csv "${TRIAD_CSV}")
fi
if [[ -n "${AI_CONTROL_CSV}" ]]; then
  BUILD_CMD+=(--ai-control-csv "${AI_CONTROL_CSV}")
fi
if [[ -n "${MEM_COMPUTE_CSV}" ]]; then
  BUILD_CMD+=(--mem-compute-csv "${MEM_COMPUTE_CSV}")
fi

echo "[run_roofline] Building roofline points..."
"${BUILD_CMD[@]}"

echo "[run_roofline] Plotting roofline..."
python3 "${PROJECT_ROOT}/analysis/plot_roofline.py" \
  --points-csv "${POINTS_CSV}" \
  --ridge-json "${RIDGE_JSON}" \
  --out "${PLOT_PNG}" \
  --title "Empirical Roofline"

echo "[run_roofline] Done."
echo "[run_roofline] Outputs:"
echo "  ${POINTS_CSV}"
echo "  ${RIDGE_JSON}"
echo "  ${PLOT_PNG}"