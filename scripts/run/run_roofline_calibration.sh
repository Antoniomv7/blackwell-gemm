#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"

CAMPAIGN_ID_ARG="${1:-}"

source_repo_env_if_present
prepare_campaign "$CAMPAIGN_ID_ARG"

OUT_DIR="$RAW_DIR/roofline_calibration"
ensure_dir "$OUT_DIR"
write_provenance "$OUT_DIR"

AI_BIN="$(resolve_bin mb10_ai_control || true)"
MIX_BIN="$(resolve_bin mb3_mem_compute_mix || true)"

if [[ -z "$AI_BIN" ]]; then
    echo "[ERROR] Could not find mb10_ai_control binary in ./bin or PATH." >&2
    exit 1
fi

if [[ -z "$MIX_BIN" ]]; then
    echo "[ERROR] Could not find mb3_mem_compute_mix binary in ./bin or PATH." >&2
    exit 1
fi

# Global calibration controls.
# Override from the environment if needed:
#   ROOFLINE_N=67108864
#   ROOFLINE_BLOCK=256
#   ROOFLINE_GRID=0
#   ROOFLINE_WARMUP=5
#   ROOFLINE_REPS=20
#   AI_SWEEP_FMAS="1 2 4 8 16 32 64 128"
#   MIX_COMPUTE_ITERS="1 2 4 8 16 32 64"
ROOFLINE_N="${ROOFLINE_N:-67108864}"
ROOFLINE_BLOCK="${ROOFLINE_BLOCK:-256}"
ROOFLINE_GRID="${ROOFLINE_GRID:-0}"
ROOFLINE_WARMUP="${ROOFLINE_WARMUP:-5}"
ROOFLINE_REPS="${ROOFLINE_REPS:-20}"
AI_SWEEP_FMAS_STR="${AI_SWEEP_FMAS:-1 2 4 8 16 32 64 128}"
MIX_COMPUTE_ITERS_STR="${MIX_COMPUTE_ITERS:-1 2 4 8 16 32 64}"

read -r -a FMAS_LIST <<< "$AI_SWEEP_FMAS_STR"
read -r -a COMPUTE_ITERS_LIST <<< "$MIX_COMPUTE_ITERS_STR"

AI_SWEEP_CSV="$OUT_DIR/ai_sweep.csv"
MIX_CSV="$OUT_DIR/mem_compute_mix.csv"
rm -f "$AI_SWEEP_CSV" "$MIX_CSV"

echo "[INFO] Running roofline calibration for campaign: $CAMPAIGN_ID"
echo "[INFO] Output directory: $OUT_DIR"
echo "[INFO] AI binary: $AI_BIN"
echo "[INFO] Mix binary: $MIX_BIN"
echo "[INFO] n=$ROOFLINE_N block=$ROOFLINE_BLOCK grid=$ROOFLINE_GRID warmup=$ROOFLINE_WARMUP reps=$ROOFLINE_REPS"
echo "[INFO] AI sweep FMAs: ${FMAS_LIST[*]}"
echo "[INFO] Mix compute iters: ${COMPUTE_ITERS_LIST[*]}"

for FMAS in "${FMAS_LIST[@]}"; do
    TAG="fmas_${FMAS}"
    TMP_CSV="$OUT_DIR/${TAG}.csv"
    LOG_FILE="$OUT_DIR/${TAG}.log"

    echo "[INFO] AI sweep run -> fmas_per_load=$FMAS"

    "$AI_BIN" \
        --n "$ROOFLINE_N" \
        --block "$ROOFLINE_BLOCK" \
        --grid "$ROOFLINE_GRID" \
        --warmup "$ROOFLINE_WARMUP" \
        --reps "$ROOFLINE_REPS" \
        --fmas-per-load "$FMAS" \
        --csv "$TMP_CSV" \
        > "$LOG_FILE" 2>&1

    append_csv "$TMP_CSV" "$AI_SWEEP_CSV"
done

for COMPUTE_ITERS in "${COMPUTE_ITERS_LIST[@]}"; do
    TAG="mix_compute_iters_${COMPUTE_ITERS}"
    TMP_CSV="$OUT_DIR/${TAG}.csv"
    LOG_FILE="$OUT_DIR/${TAG}.log"

    echo "[INFO] Mem/compute mix run -> compute_iters=$COMPUTE_ITERS"

    "$MIX_BIN" \
        --n "$ROOFLINE_N" \
        --block "$ROOFLINE_BLOCK" \
        --grid "$ROOFLINE_GRID" \
        --warmup "$ROOFLINE_WARMUP" \
        --reps "$ROOFLINE_REPS" \
        --compute-iters "$COMPUTE_ITERS" \
        --csv "$TMP_CSV" \
        > "$LOG_FILE" 2>&1

    append_csv "$TMP_CSV" "$MIX_CSV"
done

echo "[INFO] Roofline calibration completed."
echo "[INFO] AI sweep CSV -> $AI_SWEEP_CSV"
echo "[INFO] Mem/compute mix CSV -> $MIX_CSV"