#!/usr/bin/env bash
# Run script_setup → image_setup → vid_setup sequentially (native, not Docker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
RUNNER="${ROOT}/src/pipeline_runner.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ${PYTHON}. Run ./scripts/install.sh first." >&2
  exit 1
fi

# Xet downloads often fail with opaque errors when disk is full; use HTTP instead.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Prefer /scratch when $HOME is full (common on shared clusters).
if [[ -z "${HF_HOME:-}" && -d /scratch ]]; then
  export HF_HOME="/scratch/${USER}/hf_cache"
  mkdir -p "$HF_HOME" 2>/dev/null || true
fi

exec "$PYTHON" "$RUNNER" "$@"
