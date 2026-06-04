# syntax=docker/dockerfile:1.7

# RND4Impact GPU image: offline vLLM inference (script_setup) plus the full
# pyproject ML stack. Built on the pinned CUDA 12.4 / torch 2.6.0 stack so the
# compiled `vllm._C` extension is present -- that extension is what is missing
# on native Windows installs ("ModuleNotFoundError: No module named 'vllm._C'").

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
# Ubuntu 22.04 ships Python 3.10, which is within requires-python (>=3.10,<3.13).
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

# uv: fast, reproducible resolver/installer.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Force uv to use the system interpreter (never a uv-managed download) so the
# venv copied into the runtime stage keeps a valid interpreter at the same path.
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_PYTHON=python${PYTHON_VERSION} \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=600

WORKDIR /workspace

# Only the manifest is copied. The committed uv.lock is resolved on Windows (and
# is missing vllm), so we deliberately re-resolve on Linux here to pull the cu124
# vLLM/torch wheels, including the vllm._C extension.
COPY pyproject.toml ./

# Install the full dependency set without building the root project: there is no
# src/rnd4impact package, and the code runs directly from src/.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

# Fail the build early if the compiled vLLM extension did not ship. This is a
# file-level check (no import), so the NVIDIA driver is not required at build
# time -- the driver is injected at run time via `--gpus`.
RUN set -eux; \
    ls -1 /workspace/.venv/lib/python*/site-packages/vllm/_C*.so; \
    /workspace/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

############################
# Stage 2: runtime
############################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Same Python minor version as the builder so the copied venv's compiled
# extensions match the interpreter ABI. The extra libs cover the wider stack
# (opencv -> libgl1/libglib2.0-0, soundfile -> libsndfile1, audio/video -> ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/workspace/.venv/bin:${PATH}" \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Prebuilt virtualenv with the whole stack (torch, vLLM, diffusers, TTS, ...).
COPY --from=builder /workspace/.venv /workspace/.venv

# Application code and configs.
COPY src/ /workspace/src/
COPY pyproject.toml README.md LICENSE /workspace/

WORKDIR /workspace/src

# Default to the full pipeline; compose services override the command as needed.
CMD ["python", "script_setup/script_setup_runner.py", "--config", "configs/script_setup_qwen3_4b.yaml", "--all"]
