#!/usr/bin/env bash
# Sync (uv sync) and run RND4Impact pipeline setups for a VRAM tier.
#
# Order: script_setup -> image_setup -> vid_setup -> audio_setup
#
# Usage:
#   ./scripts/run-full-pipeline.sh
#   RND4IMPACT_VRAM_TIER=12gb ./scripts/run-full-pipeline.sh
#   ./scripts/run-full-pipeline.sh --tier 24gb --only script,image
#   ./scripts/run-full-pipeline.sh --skip-sync --only audio
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${RND4IMPACT_VRAM_TIER:-40gb}"
SKIP_SYNC=0
ONLY=""
EXTRA_ARGS=()

VALID_TIERS=(6gb 12gb 24gb 40gb 80gb)
ALL_SETUPS=(script_setup image_setup vid_setup audio_setup)

usage() {
  cat <<'EOF'
Usage: run-full-pipeline.sh [OPTIONS] [-- RUNNER_ARGS...]

Sync each setup with uv and run pipeline runners using matching *_<tier>.yaml configs.

Options:
  --tier TIER       VRAM tier: 6gb, 12gb, 24gb, 40gb, 80gb (default: 40gb or
                    RND4IMPACT_VRAM_TIER)
  --only SETUPS     Comma-separated subset: script,image,vid,audio (default: all)
  --skip-sync       Skip uv sync (use existing .venv deps)
  -h, --help        Show this help

Environment:
  RND4IMPACT_VRAM_TIER   Default tier when --tier is omitted

Examples:
  RND4IMPACT_VRAM_TIER=12gb ./scripts/run-full-pipeline.sh
  ./scripts/run-full-pipeline.sh --tier 6gb --only script,image,vid,audio
  ./scripts/run-full-pipeline.sh --only audio --tier 12gb
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      TIER="${2:?--tier requires a value}"
      shift 2
      ;;
    --only)
      ONLY="${2:?--only requires a value}"
      shift 2
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

tier_valid=0
for candidate in "${VALID_TIERS[@]}"; do
  if [[ "$TIER" == "$candidate" ]]; then
    tier_valid=1
    break
  fi
done
if [[ "$tier_valid" -eq 0 ]]; then
  echo "Invalid tier: $TIER (expected one of: ${VALID_TIERS[*]})" >&2
  exit 1
fi

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY="$(command -v python)"
fi

should_run() {
  local key="$1"
  if [[ -z "$ONLY" ]]; then
    return 0
  fi
  [[ ",${ONLY}," == *",${key},"* ]]
}

config_for() {
  local setup="$1"
  echo "configs/${setup}_${TIER}.yaml"
}

sync_setup() {
  local setup="$1"
  echo "=== uv sync: ${setup} ==="
  uv sync --directory "${ROOT}/src/${setup}"
}

config_path() {
  local setup="$1"
  local rel
  rel="$(config_for "${setup}")"
  local abs="${ROOT}/src/${setup}/${rel}"
  if [[ ! -f "$abs" ]]; then
    echo "Missing config for tier ${TIER}: ${abs}" >&2
    exit 1
  fi
  echo "$rel"
}

run_script_setup() {
  local cfg
  cfg="$(config_path script_setup)"
  echo "=== script_setup (${cfg}) ==="
  if [[ "$SKIP_SYNC" -eq 0 ]]; then
    sync_setup script_setup
  fi
  (
    cd "${ROOT}/src/script_setup"
    "$PY" script_setup_runner.py --config "$cfg" --all "${EXTRA_ARGS[@]}"
  )
}

run_image_setup() {
  local cfg
  cfg="$(config_path image_setup)"
  echo "=== image_setup (${cfg}) ==="
  if [[ "$SKIP_SYNC" -eq 0 ]]; then
    sync_setup image_setup
  fi
  (
    cd "${ROOT}/src/image_setup"
    "$PY" image_setup_runner.py --config "$cfg" --all "${EXTRA_ARGS[@]}"
  )
}

run_vid_setup() {
  local cfg
  cfg="$(config_path vid_setup)"
  echo "=== vid_setup (${cfg}) ==="
  if [[ "$SKIP_SYNC" -eq 0 ]]; then
    sync_setup vid_setup
  fi
  (
    cd "${ROOT}/src/vid_setup"
    "$PY" vid_setup_runner.py --config "$cfg" "${EXTRA_ARGS[@]}"
  )
}

run_audio_setup() {
  local cfg
  cfg="$(config_path audio_setup)"
  echo "=== audio_setup (${cfg}) ==="
  if [[ "$SKIP_SYNC" -eq 0 ]]; then
    sync_setup audio_setup
  fi
  (
    cd "${ROOT}/src/audio_setup"
    "$PY" audio_setup_runner.py --config "$cfg" "${EXTRA_ARGS[@]}"
  )
}

echo "RND4Impact pipeline | tier=${TIER} | python=${PY}"
if [[ -n "$ONLY" ]]; then
  echo "Setups: ${ONLY}"
else
  echo "Setups: ${ALL_SETUPS[*]}"
fi
echo "--------------------------------"

if should_run script; then
  run_script_setup
  echo "--------------------------------"
fi

if should_run image; then
  run_image_setup
  echo "--------------------------------"
fi

if should_run vid; then
  run_vid_setup
  echo "--------------------------------"
fi

if should_run audio; then
  run_audio_setup
  echo "--------------------------------"
fi

echo "Pipeline completed successfully (tier=${TIER})"
