# syntax=docker/dockerfile:1.7

# RND4Impact vid_setup GPU image: SVD / video diffusion via diffusers on CUDA 12.4.
# Dependencies resolve from src/vid_setup/pyproject.toml (not the workspace root).

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

############################
# Stage 1: builder
############################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_PYTHON=python${PYTHON_VERSION} \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=600

WORKDIR /workspace

COPY src/vid_setup/pyproject.toml src/vid_setup/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --directory src/vid_setup --no-install-project --no-dev

RUN set -eux; \
    /workspace/.venv/bin/python -c "\
import accelerate; \
import diffusers; \
import setuptools; \
import torch; \
import xformers; \
assert torch.version.cuda, 'expected CUDA-enabled torch from src/vid_setup/pyproject.toml'; \
print('setuptools', setuptools.__version__); \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('xformers', xformers.__version__); \
print('accelerate', accelerate.__version__); \
print('diffusers', diffusers.__version__); \
"

############################
# Stage 2: runtime
############################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/workspace/.venv/bin:${PATH}" \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

COPY --from=builder /workspace/.venv /workspace/.venv

COPY src/vid_setup/ /workspace/src/vid_setup/
COPY README.md LICENSE /workspace/

WORKDIR /workspace/src

CMD ["python", "vid_setup/vid_setup_runner.py", "--config", "configs/vid_setup_svd.yaml", "--all"]
