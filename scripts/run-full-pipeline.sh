#!/usr/bin/env bash
# Run script_setup -> image_setup -> vid_setup sequentially (native Python, no Docker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${RND4IMPACT_VRAM_TIER:-40gb}"
PY="${ROOT}/.venv/bin/python"

cd "$ROOT"
uv sync --directory src/script_setup
"$PY" "${ROOT}/src/script_setup/script_setup_runner.py" \
  --config "configs/script_setup_${TIER}.yaml" --all
uv sync --directory src/image_setup
"$PY" "${ROOT}/src/image_setup/image_setup_runner.py" \
  --config "configs/image_setup_${TIER}.yaml" --all
uv sync --directory src/vid_setup
"$PY" "${ROOT}/src/vid_setup/vid_setup_runner.py" \
  --config "configs/vid_setup_${TIER}.yaml"
