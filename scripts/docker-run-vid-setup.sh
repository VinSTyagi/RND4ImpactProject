#!/usr/bin/env bash
# Build (optional) and run vid_setup in Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="$ROOT/docker/vid_setup"
CONFIG="${VID_SETUP_CONFIG:-configs/vid_setup_svd.yaml}"
BUILD=0
RUNNER_ARGS=()

usage() {
  cat <<'EOF'
Usage: docker-run-vid-setup.sh [options] [-- extra runner args]

Run vid_setup in Docker. Video diffusion weights load from Hugging Face on first
run and are cached in the `rnd4impact_vid_hf_cache` Docker volume.

Options:
  --build              Run docker compose build before starting the container
  --config PATH        YAML config inside the container (default: configs/vid_setup_svd.yaml)
  --all                Run all stages (default)
  -h, --help           Show this help

Environment:
  VID_SETUP_CONFIG     Default config path (same as --config)

Examples:
  ./scripts/docker-run-vid-setup.sh --build
  ./scripts/docker-run-vid-setup.sh --config configs/vid_setup_svd.yaml --all
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
    --all)
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

exec docker compose run --rm vid-setup \
  python vid_setup/vid_setup_runner.py \
  --config "$CONFIG" \
  "${RUNNER_ARGS[@]}"
