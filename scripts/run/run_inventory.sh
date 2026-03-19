#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"

CAMPAIGN_ID_ARG="${1:-}"

source_repo_env_if_present
prepare_campaign "$CAMPAIGN_ID_ARG"

OUT_DIR="$RAW_DIR/system"
ensure_dir "$OUT_DIR"
write_provenance "$OUT_DIR"

PHYSICAL_DEVICE="${BLACKWELL_GPU_INDEX:-0}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
LOGICAL_DEVICE=0

# Limpieza de artefacto legacy
rm -f "$OUT_DIR/selected_device.txt"

echo "$PHYSICAL_DEVICE" > "$OUT_DIR/selected_physical_device.txt"
echo "$LOGICAL_DEVICE" > "$OUT_DIR/selected_logical_device.txt"

echo "[INFO] Running inventory for campaign: $CAMPAIGN_ID"
echo "[INFO] Output directory: $OUT_DIR"
echo "[INFO] Physical GPU selected via CUDA_VISIBLE_DEVICES: $PHYSICAL_DEVICE"
echo "[INFO] Logical CUDA device passed to mb31_inventory: $LOGICAL_DEVICE"

INVENTORY_BIN="$(resolve_bin mb31_inventory || true)"
if [[ -z "$INVENTORY_BIN" ]]; then
    echo "[ERROR] Could not find mb31_inventory binary in ./bin or PATH." >&2
    exit 1
fi

if [[ -f "$ROOT_DIR/scripts/collect_env.sh" ]]; then
    bash "$ROOT_DIR/scripts/collect_env.sh" "$OUT_DIR/collect_env.txt" \
        > "$OUT_DIR/collect_env.log" 2>&1
fi

if [[ -f "$ROOT_DIR/benchmarks/device/mb31_topology.sh" ]]; then
    bash "$ROOT_DIR/benchmarks/device/mb31_topology.sh" \
        > "$OUT_DIR/topology.txt" 2>&1 || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi topo -m > "$OUT_DIR/topo_matrix.txt" 2>&1 || true
    nvidia-smi -L > "$OUT_DIR/gpu_list.txt" 2>&1 || true
fi

"$INVENTORY_BIN" \
    --device "$LOGICAL_DEVICE" \
    --csv "$OUT_DIR/device_inventory.csv" \
    --txt "$OUT_DIR/device_inventory.txt" \
    > "$OUT_DIR/device_inventory.log" 2>&1

echo "[INFO] Inventory completed."
echo "[INFO] device_inventory.csv -> $OUT_DIR/device_inventory.csv"
echo "[INFO] device_inventory.txt -> $OUT_DIR/device_inventory.txt"