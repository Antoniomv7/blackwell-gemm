#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"

OUTPUT_CAMPAIGN_ID="${1:-}"

source_repo_env_if_present
prepare_campaign "$OUTPUT_CAMPAIGN_ID"

HBM_SOURCE_CAMPAIGN="${HBM_SOURCE_CAMPAIGN:-$CAMPAIGN_ID}"
CALIB_SOURCE_CAMPAIGN="${CALIB_SOURCE_CAMPAIGN:-$CAMPAIGN_ID}"

HBM_CSV="$ROOT_DIR/results/raw/$HBM_SOURCE_CAMPAIGN/hbm/hbm_stream_triad.csv"
AI_CSV="$ROOT_DIR/results/raw/$CALIB_SOURCE_CAMPAIGN/roofline_calibration/ai_sweep.csv"
MIX_CSV="$ROOT_DIR/results/raw/$CALIB_SOURCE_CAMPAIGN/roofline_calibration/mem_compute_mix.csv"

ROOFLINE_DIR="$PROCESSED_DIR/roofline"
FIGURES_DIR="$PROCESSED_DIR/figures"
TABLES_DIR="$PROCESSED_DIR/tables"

ensure_dir "$ROOFLINE_DIR"
ensure_dir "$FIGURES_DIR"
ensure_dir "$TABLES_DIR"
write_provenance "$PROCESSED_DIR"

require_cmd python3

echo "[INFO] Output campaign: $CAMPAIGN_ID"
echo "[INFO] HBM source campaign: $HBM_SOURCE_CAMPAIGN"
echo "[INFO] Calibration source campaign: $CALIB_SOURCE_CAMPAIGN"
echo "[INFO] HBM CSV: $HBM_CSV"
echo "[INFO] AI CSV: $AI_CSV"
echo "[INFO] MIX CSV: $MIX_CSV"

python3 "$ROOT_DIR/analysis/build_empirical_roofline.py" \
  --hbm-csv "$HBM_CSV" \
  --ai-csv "$AI_CSV" \
  --mix-csv "$MIX_CSV" \
  --processed-campaign-dir "$PROCESSED_DIR"

python3 "$ROOT_DIR/analysis/plot_roofline.py" \
  --points-csv "$ROOFLINE_DIR/roofline_points.csv" \
  --ridge-json "$ROOFLINE_DIR/empirical_roofline.json" \
  --out "$FIGURES_DIR/roofline.png"

python3 "$ROOT_DIR/analysis/summarize_campaign.py" \
  --processed-campaign-dir "$PROCESSED_DIR" \
  --output "$TABLES_DIR/campaign_summary.txt"

echo "[INFO] Roofline analysis completed for campaign: $CAMPAIGN_ID"