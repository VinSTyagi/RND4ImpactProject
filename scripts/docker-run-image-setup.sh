#!/usr/bin/env bash
# Build (optional) and run image_setup in Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="$ROOT/docker/image_setup"
CONFIG="${IMAGE_SETUP_CONFIG:-configs/image_setup_sdxl_fp16.yaml}"
BUILD=0
RUNNER_ARGS=()

usage() {
  cat <<'EOF'
Usage: docker-run-image-setup.sh [options] [-- extra runner args]

Run image_setup in Docker. SDXL weights load from Hugging Face on first run
and are cached in the `rnd4impact_image_hf_cache` Docker volume.

Options:
  --build              Run docker compose build before starting the container
  --config PATH        YAML config inside the container (default: configs/image_setup_sdxl_fp16.yaml)
  --all                Run all stages (default)
  --1, --2             Run only stage 1 or 2
  -h, --help           Show this help

Environment:
  IMAGE_SETUP_CONFIG   Default config path (same as --config)

Examples:
  ./scripts/docker-run-image-setup.sh --build
  ./scripts/docker-run-image-setup.sh --config configs/image_setup_sdxl_low_vram.yaml --all
  ./scripts/docker-run-image-setup.sh --1
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
    --all|--1|--2)
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

exec docker compose run --rm image-setup \
  python image_setup/image_setup_runner.py \
  --config "$CONFIG" \
  "${RUNNER_ARGS[@]}"
