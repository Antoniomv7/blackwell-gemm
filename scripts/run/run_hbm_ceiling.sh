#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"

CAMPAIGN_ID_ARG="${1:-}"

source_repo_env_if_present
prepare_campaign "$CAMPAIGN_ID_ARG"

OUT_DIR="$RAW_DIR/hbm"
ensure_dir "$OUT_DIR"
write_provenance "$OUT_DIR"

HBM_BIN="$(resolve_bin mb34_stream_triad || true)"
if [[ -z "$HBM_BIN" ]]; then
    echo "[ERROR] Could not find mb34_stream_triad binary in ./bin or PATH." >&2
    exit 1
fi

# You can override these through the environment if needed:
#   HBM_SIZES="67108864 134217728"
#   HBM_BLOCKS="256 512"
#   HBM_WARMUP=5
#   HBM_REPS=20
#   HBM_GRID=0
HBM_SIZES_STR="${HBM_SIZES:-67108864 134217728}"
HBM_BLOCKS_STR="${HBM_BLOCKS:-256 512}"
HBM_WARMUP="${HBM_WARMUP:-5}"
HBM_REPS="${HBM_REPS:-20}"
HBM_GRID="${HBM_GRID:-0}"

read -r -a SIZES <<< "$HBM_SIZES_STR"
read -r -a BLOCKS <<< "$HBM_BLOCKS_STR"

AGG_CSV="$OUT_DIR/hbm_stream_triad.csv"
rm -f "$AGG_CSV"

echo "[INFO] Running HBM ceiling for campaign: $CAMPAIGN_ID"
echo "[INFO] Output directory: $OUT_DIR"
echo "[INFO] Binary: $HBM_BIN"
echo "[INFO] Sizes: ${SIZES[*]}"
echo "[INFO] Blocks: ${BLOCKS[*]}"
echo "[INFO] Warmup: $HBM_WARMUP"
echo "[INFO] Reps: $HBM_REPS"
echo "[INFO] Grid: $HBM_GRID"

for N in "${SIZES[@]}"; do
    for BLOCK in "${BLOCKS[@]}"; do
        RUN_TAG="n${N}_b${BLOCK}"
        TMP_CSV="$OUT_DIR/${RUN_TAG}.csv"
        LOG_FILE="$OUT_DIR/${RUN_TAG}.log"

        echo "[INFO] HBM run -> n=$N block=$BLOCK"

        "$HBM_BIN" \
            --n "$N" \
            --block "$BLOCK" \
            --grid "$HBM_GRID" \
            --warmup "$HBM_WARMUP" \
            --reps "$HBM_REPS" \
            --csv "$TMP_CSV" \
            > "$LOG_FILE" 2>&1

        append_csv "$TMP_CSV" "$AGG_CSV"
    done
done

echo "[INFO] HBM ceiling completed."
echo "[INFO] Aggregate CSV -> $AGG_CSV"