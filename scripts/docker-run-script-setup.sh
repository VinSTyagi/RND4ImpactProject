#!/usr/bin/env bash
# Build (optional) and run script_setup in Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="$ROOT/docker/script_setup"
CONFIG="${SCRIPT_SETUP_CONFIG:-configs/script_setup_qwen3_4b.yaml}"
BUILD=0
RUNNER_ARGS=()

usage() {
  cat <<'EOF'
Usage: docker-run-script-setup.sh [options] [-- extra runner args]

Run script_setup in Docker. LLM weights load from Hugging Face on first run
and are cached in the compose hf_cache volume (rnd4impact_script_hf_cache).

Options:
  --build              Run docker compose build before starting the container
  --config PATH        YAML config inside the container (default: configs/script_setup_qwen3_4b.yaml)
  --all                Run all stages (default)
  --1, --2, --3, --4   Run only the given stage
  -h, --help           Show this help

Environment:
  SCRIPT_SETUP_CONFIG   Default config path (same as --config)

Examples:
  ./scripts/docker-run-script-setup.sh --build
  ./scripts/docker-run-script-setup.sh --config configs/script_setup_qwen3_4b.yaml --all
  ./scripts/docker-run-script-setup.sh --4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD=1
      shift
      ;;
    --config)
      CONFIG="${2:?missing value for --config}"
      shift 2
      ;;
    --all|--1|--2|--3|--4)
      RUNNER_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      RUNNER_ARGS+=("$@")
      break
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#RUNNER_ARGS[@]} -eq 0 ]]; then
  RUNNER_ARGS=(--all)
fi

cd "$COMPOSE_DIR"

if [[ "$BUILD" -eq 1 ]]; then
  docker compose build
fi

exec docker compose run --rm script-setup \
  python script_setup/script_setup_runner.py \
  --config "$CONFIG" \
  "${RUNNER_ARGS[@]}"
