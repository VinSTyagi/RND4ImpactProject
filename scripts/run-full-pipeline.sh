#!/usr/bin/env bash
# Run script_setup -> image_setup -> vid_setup sequentially (native Python, no Docker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${RND4IMPACT_VRAM_TIER:-24b}"
PY="${ROOT}/.venv/bin/python"

cd "$ROOT"
echo "Syncing script_setup"
uv sync --directory src/script_setup
"$PY" "${ROOT}/src/script_setup/script_setup_runner.py" \
  --config "configs/script_setup_6gb.yaml" --all
echo "--------------------------------"
echo "Syncing image_setup"
uv sync --directory src/image_setup
"$PY" "${ROOT}/src/image_setup/image_setup_runner.py" \
  --config "configs/image_setup_${TIER}.yaml" --all
echo "--------------------------------"
echo "Syncing vid_setup"
uv sync --directory src/vid_setup
"$PY" "${ROOT}/src/vid_setup/vid_setup_runner.py" \
  --config "configs/vid_setup_${TIER}.yaml"
echo "--------------------------------"
# echo "Syncing audio_setup"
# uv sync --directory src/audio_setup
# "$PY" "${ROOT}/src/audio_setup/audio_setup_runner.py" \
#   --config "configs/audio_setup_${TIER}.yaml" --all
echo "--------------------------------"
echo "Full pipeline completed successfully"