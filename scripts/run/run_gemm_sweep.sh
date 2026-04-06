#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"

usage() {
    cat <<'EOF'
Usage:
  run_gemm_sweep.sh [campaign_id] [config_path] [--dry-run]

Arguments:
  campaign_id   Optional campaign ID
  config_path   Optional config JSON path. Default: configs/gemm_shapes_mvp.json

Options:
  --dry-run     Expand the sweep and print planned cases without calling the GEMM binary

Examples:
  bash scripts/run/run_gemm_sweep.sh test_gemm
  bash scripts/run/run_gemm_sweep.sh test_gemm configs/gemm_shapes_mvp.json --dry-run
EOF
}

CAMPAIGN_ID_ARG=""
CONFIG_PATH=""
DRY_RUN=0

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) POSITIONAL+=("$1"); shift 1 ;;
    esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
    CAMPAIGN_ID_ARG="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
    CONFIG_PATH="${POSITIONAL[1]}"
fi
if [[ ${#POSITIONAL[@]} -gt 2 ]]; then
    echo "[ERROR] Too many positional arguments." >&2
    usage
    exit 1
fi

if [[ -z "$CONFIG_PATH" ]]; then
    CONFIG_PATH="$ROOT_DIR/configs/gemm_shapes_mvp.json"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] GEMM config not found: $CONFIG_PATH" >&2
    exit 1
fi

source_repo_env_if_present
prepare_campaign "$CAMPAIGN_ID_ARG"

OUT_DIR="$RAW_DIR/gemm"
ensure_dir "$OUT_DIR"
write_provenance "$OUT_DIR"

require_cmd python3

AGG_CSV="$OUT_DIR/gemm_sweep.csv"
if [[ "$DRY_RUN" -eq 0 ]]; then
    rm -f "$AGG_CSV"
fi

PLANNED_CASES=0

while IFS=$'\t' read -r CASE_ID FAMILY M N K DTYPE TRANSA TRANSB WORKSPACE WARMUP REPS; do
    [[ -z "${CASE_ID:-}" ]] && continue
    PLANNED_CASES=$((PLANNED_CASES + 1))

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] case_id=$CASE_ID family=$FAMILY m=$M n=$N k=$K dtype=$DTYPE layout=${TRANSA}${TRANSB} workspace=$WORKSPACE warmup=$WARMUP reps=$REPS"
    else
        "$SCRIPT_DIR/run_gemm_case.sh" \
            --campaign-id "$CAMPAIGN_ID" \
            --case-id "$CASE_ID" \
            --family "$FAMILY" \
            --m "$M" \
            --n "$N" \
            --k "$K" \
            --dtype "$DTYPE" \
            --transa "$TRANSA" \
            --transb "$TRANSB" \
            --workspace-bytes "$WORKSPACE" \
            --warmup "$WARMUP" \
            --reps "$REPS" \
            --append-csv "$AGG_CSV"
    fi
done < <(
python3 - "$CONFIG_PATH" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = json.load(f)

default_warmup = cfg.get("warmup", 10)
default_reps = cfg.get("reps", 30)

for case in cfg["cases"]:
    case_id = case["case_id"]
    family = case["family"]
    m = case["m"]
    n = case["n"]
    k = case["k"]
    warmup = case.get("warmup", default_warmup)
    reps = case.get("reps", default_reps)

    for dtype in case["dtypes"]:
        for layout in case["layouts"]:
            if len(layout) != 2:
                raise ValueError(f"Invalid layout '{layout}' in case '{case_id}'")
            transa, transb = layout[0], layout[1]
            for workspace in case["workspaces"]:
                print(
                    "\t".join([
                        str(case_id),
                        str(family),
                        str(m),
                        str(n),
                        str(k),
                        str(dtype),
                        str(transa),
                        str(transb),
                        str(workspace),
                        str(warmup),
                        str(reps),
                    ])
                )
PY
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[INFO] GEMM sweep dry-run completed."
    echo "[INFO] planned_cases=$PLANNED_CASES"
else
    echo "[INFO] GEMM sweep completed."
    echo "[INFO] planned_cases=$PLANNED_CASES"
    echo "[INFO] Aggregate CSV -> $AGG_CSV"
fi