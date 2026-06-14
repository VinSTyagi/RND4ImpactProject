# syntax=docker/dockerfile:1.7

# RND4Impact image_setup GPU image: SDXL inference via diffusers on CUDA 12.4.

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

COPY src/image_setup/pyproject.toml ./pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

RUN /workspace/.venv/bin/python -c "import setuptools; import torch; print('setuptools', setuptools.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"

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
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/workspace/.venv/bin:${PATH}" \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

COPY --from=builder /workspace/.venv /workspace/.venv

COPY src/image_setup/ /workspace/src/image_setup/
COPY README.md LICENSE /workspace/

WORKDIR /workspace/src

CMD ["python", "image_setup/image_setup_runner.py", "--config", "configs/image_setup_sdxl_fp16.yaml", "--all"]
