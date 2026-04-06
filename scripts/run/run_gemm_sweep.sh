#!/usr/bin/env bash
set -euo pipefail

RUN_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$RUN_SCRIPT_DIR/../lib/common.sh"

CAMPAIGN_ID_ARG="${1:-}"
CONFIG_PATH="${2:-$ROOT_DIR/configs/gemm_shapes_mvp.json}"

source_repo_env_if_present
prepare_campaign "$CAMPAIGN_ID_ARG"

OUT_DIR="$RAW_DIR/gemm"
ensure_dir "$OUT_DIR"
write_provenance "$OUT_DIR"

require_cmd python3

AGG_CSV="$OUT_DIR/gemm_sweep.csv"
rm -f "$AGG_CSV"

python3 - "$CONFIG_PATH" <<'PY' | while IFS=$'\t' read -r CASE_ID FAMILY M N K DTYPE TRANSA TRANSB WORKSPACE WARMUP REPS; do
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
    bash "$RUN_SCRIPT_DIR/run_gemm_case.sh" \
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
done

echo "[INFO] GEMM sweep completed."
echo "[INFO] Aggregate CSV -> $AGG_CSV"