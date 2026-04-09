#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"

CAMPAIGN_ID_ARG="${1:-}"
CONFIG_PATH="${2:-$ROOT_DIR/configs/ncu_cases.json}"

source_repo_env_if_present
prepare_campaign "$CAMPAIGN_ID_ARG"

require_cmd python3
require_cmd ncu

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] NCU config not found: $CONFIG_PATH" >&2
    exit 1
fi

OUT_DIR="$RAW_DIR/ncu"
ensure_dir "$OUT_DIR"
write_provenance "$OUT_DIR"

GEMM_BIN="$(resolve_gemm_bin)"
AGG_CSV="$OUT_DIR/ncu_cases.csv"
rm -f "$AGG_CSV"

echo "[INFO] Running selected NCU GEMM cases for campaign: $CAMPAIGN_ID"
echo "[INFO] Output directory: $OUT_DIR"
echo "[INFO] GEMM binary: $GEMM_BIN"
echo "[INFO] Config: $CONFIG_PATH"

python3 - "$CONFIG_PATH" <<'PY' | while IFS=$'\t' read -r CASE_ID FAMILY M N K DTYPE TRANSA TRANSB LDA LDB LDC WORKSPACE WARMUP REPS; do
import json
import sys

cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = json.load(f)

default_warmup = cfg.get("warmup", 10)
default_reps = cfg.get("reps", 20)

for case in cfg["cases"]:
    print("\t".join([
        str(case["case_id"]),
        str(case["family"]),
        str(case["m"]),
        str(case["n"]),
        str(case["k"]),
        str(case["dtype"]),
        str(case["transa"]),
        str(case["transb"]),
        str(case.get("lda", 0)),
        str(case.get("ldb", 0)),
        str(case.get("ldc", 0)),
        str(case["workspace_bytes"]),
        str(case.get("warmup", default_warmup)),
        str(case.get("reps", default_reps)),
    ]))
PY
    CASE_CSV="$OUT_DIR/${CASE_ID}.csv"
    STDOUT_LOG="$OUT_DIR/${CASE_ID}.stdout.txt"
    STDERR_LOG="$OUT_DIR/${CASE_ID}.stderr.txt"
    REP_BASE="$OUT_DIR/${CASE_ID}"

    rm -f "$CASE_CSV" "$STDOUT_LOG" "$STDERR_LOG" "${REP_BASE}.ncu-rep"

    echo "[INFO] Profiling case: $CASE_ID"
    echo "[INFO]   family=$FAMILY m=$M n=$N k=$K dtype=$DTYPE layout=${TRANSA}${TRANSB} workspace=$WORKSPACE"

    set +e
    ncu \
        --target-processes all \
        --force-overwrite true \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section LaunchStats \
        --section SchedulerStats \
        -o "$REP_BASE" \
        "$GEMM_BIN" \
            --case-id "$CASE_ID" \
            --family "$FAMILY" \
            --m "$M" \
            --n "$N" \
            --k "$K" \
            --dtype "$DTYPE" \
            --transa "$TRANSA" \
            --transb "$TRANSB" \
            --lda "$LDA" \
            --ldb "$LDB" \
            --ldc "$LDC" \
            --workspace-bytes "$WORKSPACE" \
            --warmup "$WARMUP" \
            --reps "$REPS" \
            --csv "$CASE_CSV" \
        >"$STDOUT_LOG" 2>"$STDERR_LOG"
    NCU_STATUS=$?
    set -e

    if [[ $NCU_STATUS -ne 0 ]]; then
        echo "[ERROR] ncu failed for case: $CASE_ID" >&2
        echo "[ERROR] ncu exit code: $NCU_STATUS" >&2
        echo "[ERROR] stderr tail:" >&2
        tail -n 80 "$STDERR_LOG" >&2 || true
        exit "$NCU_STATUS"
    fi

    if [[ ! -f "$CASE_CSV" ]]; then
        echo "[ERROR] Expected CSV not found for case: $CASE_ID" >&2
        echo "[ERROR] Missing file: $CASE_CSV" >&2
        echo "[ERROR] stderr tail:" >&2
        tail -n 80 "$STDERR_LOG" >&2 || true
        echo "[ERROR] stdout tail:" >&2
        tail -n 40 "$STDOUT_LOG" >&2 || true
        exit 1
    fi

    if [[ ! -f "${REP_BASE}.ncu-rep" ]]; then
        echo "[ERROR] Expected NCU report not found for case: $CASE_ID" >&2
        echo "[ERROR] Missing file: ${REP_BASE}.ncu-rep" >&2
        exit 1
    fi

    append_csv "$CASE_CSV" "$AGG_CSV"
done

echo "[INFO] Selected NCU GEMM cases completed."
echo "[INFO] Aggregate CSV -> $AGG_CSV"
