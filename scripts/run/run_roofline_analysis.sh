#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"

CAMPAIGN_ID_ARG="${1:-}"

source_repo_env_if_present
prepare_campaign "$CAMPAIGN_ID_ARG"

ROOFLINE_DIR="$PROCESSED_DIR/roofline"
FIGURES_DIR="$PROCESSED_DIR/figures"
TABLES_DIR="$PROCESSED_DIR/tables"
HEATMAP_DIR="$FIGURES_DIR/gemm_heatmaps"

ensure_dir "$ROOFLINE_DIR"
ensure_dir "$FIGURES_DIR"
ensure_dir "$TABLES_DIR"
ensure_dir "$HEATMAP_DIR"

require_cmd python3

python3 "$ROOT_DIR/analysis/build_empirical_roofline.py" \
    --raw-campaign-dir "$RAW_DIR" \
    --processed-campaign-dir "$PROCESSED_DIR"

python3 "$ROOT_DIR/analysis/plot_roofline.py" \
    --roofline-points "$ROOFLINE_DIR/roofline_points.csv" \
    --roofline-summary "$ROOFLINE_DIR/empirical_roofline.json" \
    --output "$FIGURES_DIR/roofline.png"

if [[ -f "$RAW_DIR/gemm/gemm_sweep.csv" ]]; then
    python3 "$ROOT_DIR/analysis/summarize_gemm.py" \
        --gemm-csv "$RAW_DIR/gemm/gemm_sweep.csv" \
        --output "$TABLES_DIR/gemm_summary.csv"

    python3 "$ROOT_DIR/analysis/plot_gemm_heatmaps.py" \
        --gemm-csv "$RAW_DIR/gemm/gemm_sweep.csv" \
        --output-dir "$HEATMAP_DIR"
fi

python3 "$ROOT_DIR/analysis/summarize_campaign.py" \
    --processed-campaign-dir "$PROCESSED_DIR" \
    --output "$TABLES_DIR/campaign_summary.txt"

echo "[INFO] Roofline analysis completed for campaign: $CAMPAIGN_ID"