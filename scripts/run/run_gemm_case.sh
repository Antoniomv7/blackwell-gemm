#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"

usage() {
    cat <<'EOF'
Usage:
  run_gemm_case.sh [options]

Required:
  --case-id <str>
  --family <str>
  --m <int>
  --n <int>
  --k <int>
  --dtype <fp16|bf16>

Optional:
  --campaign-id <str>
  --transa <N|T>                default: N
  --transb <N|T>                default: N
  --lda <int>                   default: 0
  --ldb <int>                   default: 0
  --ldc <int>                   default: 0
  --workspace-bytes <int>       default: 0
  --warmup <int>                default: 10
  --reps <int>                  default: 30
  --append-csv <path>
  --dry-run

Contract expected from GEMM binary:
  --case-id
  --family
  --m --n --k
  --dtype
  --transa --transb
  --lda --ldb --ldc
  --workspace-bytes
  --warmup --reps
  --csv
EOF
}

CAMPAIGN_ID_ARG=""
CASE_ID=""
FAMILY=""
M=""
N=""
K=""
DTYPE=""
TRANSA="N"
TRANSB="N"
LDA=0
LDB=0
LDC=0
WORKSPACE_BYTES=0
WARMUP=10
REPS=30
APPEND_CSV=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-id) CAMPAIGN_ID_ARG="$2"; shift 2 ;;
        --case-id) CASE_ID="$2"; shift 2 ;;
        --family) FAMILY="$2"; shift 2 ;;
        --m) M="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --transa) TRANSA="$2"; shift 2 ;;
        --transb) TRANSB="$2"; shift 2 ;;
        --lda) LDA="$2"; shift 2 ;;
        --ldb) LDB="$2"; shift 2 ;;
        --ldc) LDC="$2"; shift 2 ;;
        --workspace-bytes) WORKSPACE_BYTES="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --append-csv) APPEND_CSV="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$CASE_ID" || -z "$FAMILY" || -z "$M" || -z "$N" || -z "$K" || -z "$DTYPE" ]]; then
    echo "[ERROR] Missing required GEMM case arguments." >&2
    usage
    exit 1
fi

if [[ "$DTYPE" != "fp16" && "$DTYPE" != "bf16" ]]; then
    echo "[ERROR] Unsupported dtype: $DTYPE" >&2
    exit 1
fi

if [[ "$TRANSA" != "N" && "$TRANSA" != "T" ]]; then
    echo "[ERROR] Unsupported transa: $TRANSA" >&2
    exit 1
fi

if [[ "$TRANSB" != "N" && "$TRANSB" != "T" ]]; then
    echo "[ERROR] Unsupported transb: $TRANSB" >&2
    exit 1
fi

source_repo_env_if_present
prepare_campaign "$CAMPAIGN_ID_ARG"

OUT_DIR="$RAW_DIR/gemm"
ensure_dir "$OUT_DIR"
write_provenance "$OUT_DIR"

LAYOUT="${TRANSA}${TRANSB}"
CASE_TAG="${CASE_ID}_${DTYPE}_${LAYOUT}_ws${WORKSPACE_BYTES}"
TMP_CSV="$OUT_DIR/${CASE_TAG}.csv"
LOG_FILE="$OUT_DIR/${CASE_TAG}.log"

if [[ "$DRY_RUN" -eq 1 ]]; then
    cat <<EOF
[DRY-RUN] GEMM case contract
campaign_id=$CAMPAIGN_ID
case_id=$CASE_ID
family=$FAMILY
m=$M
n=$N
k=$K
dtype=$DTYPE
transa=$TRANSA
transb=$TRANSB
layout=$LAYOUT
lda=$LDA
ldb=$LDB
ldc=$LDC
workspace_bytes=$WORKSPACE_BYTES
warmup=$WARMUP
reps=$REPS
tmp_csv=$TMP_CSV
append_csv=${APPEND_CSV:-<none>}
EOF
    exit 0
fi

GEMM_BIN="$(resolve_gemm_bin)"
echo "[INFO] GEMM case: $CASE_TAG"
echo "[INFO] Using GEMM binary: $GEMM_BIN"

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
    --workspace-bytes "$WORKSPACE_BYTES" \
    --warmup "$WARMUP" \
    --reps "$REPS" \
    --csv "$TMP_CSV" \
    > "$LOG_FILE" 2>&1

if [[ -n "$APPEND_CSV" ]]; then
    append_csv "$TMP_CSV" "$APPEND_CSV"
fi

echo "[INFO] GEMM case completed -> $TMP_CSV"