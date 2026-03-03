#!/usr/bin/env bash
set -euo pipefail

echo "[post_create] Python:"
python -V

echo "[post_create] Micromamba envs:"
micromamba env list || true

echo "[post_create] Git config (safe defaults):"
git config --global init.defaultBranch main
git config --global pull.rebase false

echo "[post_create] Done."