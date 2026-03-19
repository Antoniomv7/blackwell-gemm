#!/usr/bin/env bash
set -euo pipefail

THIS_FILE="${BASH_SOURCE[0]}"
LIB_DIR="$(cd "$(dirname "$THIS_FILE")" && pwd)"
ROOT_DIR="$(cd "$LIB_DIR/../.." && pwd)"

timestamp() {
    date +"%Y%m%d_%H%M%S"
}

default_campaign_id() {
    echo "campaign_$(timestamp)"
}

prepare_campaign() {
    local cid="${1:-${CAMPAIGN_ID:-$(default_campaign_id)}}"
    export CAMPAIGN_ID="$cid"
    export ROOT_DIR
    export RAW_DIR="$ROOT_DIR/results/raw/$CAMPAIGN_ID"
    export PROCESSED_DIR="$ROOT_DIR/results/processed/$CAMPAIGN_ID"
    mkdir -p "$RAW_DIR" "$PROCESSED_DIR"
}

ensure_dir() {
    mkdir -p "$1"
}

source_repo_env_if_present() {
    if [[ -f "$ROOT_DIR/scripts/env/env.sh" ]]; then
        # shellcheck source=/dev/null
        source "$ROOT_DIR/scripts/env/env.sh"
    fi
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "[ERROR] Required command not found: $cmd" >&2
        exit 1
    }
}

resolve_bin() {
    local name="$1"

    if [[ -x "$ROOT_DIR/bin/$name" ]]; then
        echo "$ROOT_DIR/bin/$name"
        return 0
    fi

    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi

    return 1
}

resolve_gemm_bin() {
    local candidate=""
    for candidate in "cublaslt_gemm_sweep" "mb4_mb5_cublaslt_baseline"; do
        if resolve_bin "$candidate" >/dev/null 2>&1; then
            resolve_bin "$candidate"
            return 0
        fi
    done

    echo "[ERROR] Could not resolve GEMM binary. Expected one of: cublaslt_gemm_sweep, mb4_mb5_cublaslt_baseline" >&2
    exit 1
}

append_csv() {
    local src="$1"
    local dst="$2"

    python3 - "$src" "$dst" <<'PY'
import csv
import os
import sys

src, dst = sys.argv[1], sys.argv[2]

if not os.path.exists(src):
    raise SystemExit(f"Source CSV does not exist: {src}")

with open(src, newline="") as f:
    rows = list(csv.reader(f))

if not rows:
    raise SystemExit(f"Source CSV is empty: {src}")

header, data = rows[0], rows[1:]
dst_exists = os.path.exists(dst) and os.path.getsize(dst) > 0

with open(dst, "a", newline="") as f:
    writer = csv.writer(f)
    if not dst_exists:
        writer.writerow(header)
    writer.writerows(data)
PY
}

write_provenance() {
    local outdir="$1"
    ensure_dir "$outdir"

    {
        echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
        echo "campaign_id=${CAMPAIGN_ID:-unset}"
        echo "root_dir=$ROOT_DIR"
        echo "hostname=$(hostname)"
        echo "pwd=$(pwd)"
        echo "user=${USER:-unknown}"
        echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "blackwell_gpu_index=${BLACKWELL_GPU_INDEX:-unset}"
        echo "git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "git_status_begin"
        git -C "$ROOT_DIR" status --short 2>/dev/null || true
        echo "git_status_end"
    } > "$outdir/provenance.txt"

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi > "$outdir/nvidia_smi.txt" 2>&1 || true
        nvidia-smi -q > "$outdir/nvidia_smi_q.txt" 2>&1 || true
    fi

    env | sort > "$outdir/environment.txt"
}

log_cmd() {
    local logfile="$1"
    shift

    {
        echo "[CMD] $*"
        "$@"
    } | tee "$logfile"
}