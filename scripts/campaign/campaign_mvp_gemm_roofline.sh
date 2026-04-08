#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$ROOT_DIR/scripts/run"

usage() {
    cat <<EOF
Usage:
  bash scripts/campaign/campaign_mvp_gemm_roofline.sh [campaign_id] [gemm_config]

Arguments:
  campaign_id   Optional campaign identifier. If omitted, a timestamped one is generated.
  gemm_config   Optional path to GEMM sweep config JSON.
                Default: $ROOT_DIR/configs/gemm_shapes_mvp.json

Examples:
  bash scripts/campaign/campaign_mvp_gemm_roofline.sh
  bash scripts/campaign/campaign_mvp_gemm_roofline.sh test_full
  bash scripts/campaign/campaign_mvp_gemm_roofline.sh test_full configs/gemm_shapes_mvp.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CAMPAIGN_ID="${1:-campaign_$(date +%Y%m%d_%H%M%S)}"
GEMM_CONFIG="${2:-$ROOT_DIR/configs/gemm_shapes_mvp.json}"

if [[ ! -f "$GEMM_CONFIG" ]]; then
    echo "[ERROR] GEMM config not found: $GEMM_CONFIG" >&2
    exit 1
fi

required_scripts=(
    "$RUN_DIR/run_inventory.sh"
    "$RUN_DIR/run_hbm_ceiling.sh"
    "$RUN_DIR/run_roofline_calibration.sh"
    "$RUN_DIR/run_gemm_sweep.sh"
    "$RUN_DIR/run_roofline_analysis.sh"
)

for s in "${required_scripts[@]}"; do
    if [[ ! -f "$s" ]]; then
        echo "[ERROR] Required script not found: $s" >&2
        exit 1
    fi
done

CURRENT_STEP="initialization"

on_error() {
    local exit_code=$?
    echo
    echo "[ERROR] Campaign failed."
    echo "[ERROR] campaign_id: $CAMPAIGN_ID"
    echo "[ERROR] failed_step: $CURRENT_STEP"
    echo "[ERROR] exit_code: $exit_code"
    echo "[ERROR] raw_dir: $ROOT_DIR/results/raw/$CAMPAIGN_ID"
    echo "[ERROR] processed_dir: $ROOT_DIR/results/processed/$CAMPAIGN_ID"
    exit "$exit_code"
}

trap on_error ERR

run_step() {
    local step_name="$1"
    shift

    CURRENT_STEP="$step_name"

    echo
    echo "============================================================"
    echo "[INFO] Campaign: $CAMPAIGN_ID"
    echo "[INFO] Step: $step_name"
    echo "[INFO] Command: $*"
    echo "============================================================"
    echo

    "$@"

    echo
    echo "[INFO] Completed step: $step_name"
}

echo "[INFO] Starting MVP GEMM + roofline campaign"
echo "[INFO] campaign_id: $CAMPAIGN_ID"
echo "[INFO] gemm_config: $GEMM_CONFIG"
echo "[INFO] root_dir: $ROOT_DIR"

run_step "inventory" \
    bash "$RUN_DIR/run_inventory.sh" "$CAMPAIGN_ID"

run_step "hbm_ceiling" \
    bash "$RUN_DIR/run_hbm_ceiling.sh" "$CAMPAIGN_ID"

run_step "roofline_calibration" \
    bash "$RUN_DIR/run_roofline_calibration.sh" "$CAMPAIGN_ID"

run_step "gemm_sweep" \
    bash "$RUN_DIR/run_gemm_sweep.sh" "$CAMPAIGN_ID" "$GEMM_CONFIG"

run_step "roofline_analysis" \
    bash "$RUN_DIR/run_roofline_analysis.sh" "$CAMPAIGN_ID"

echo
echo "============================================================"
echo "[INFO] Campaign completed successfully"
echo "[INFO] campaign_id: $CAMPAIGN_ID"
echo "[INFO] raw_results: $ROOT_DIR/results/raw/$CAMPAIGN_ID"
echo "[INFO] processed_results: $ROOT_DIR/results/processed/$CAMPAIGN_ID"
echo "============================================================"