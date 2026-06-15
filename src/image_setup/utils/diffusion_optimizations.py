from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_CUDA_BACKENDS_CONFIGURED = False


def configure_cuda_backends() -> None:
    """Enable TF32 and cuDNN autotuning for faster diffusion inference on CUDA."""
    global _CUDA_BACKENDS_CONFIGURED
    if _CUDA_BACKENDS_CONFIGURED or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    _CUDA_BACKENDS_CONFIGURED = True
    logger.debug("Configured CUDA backends (TF32 + cuDNN benchmark)")


def try_enable_xformers(pipeline: Any) -> bool:
    """Enable xformers memory-efficient attention when the package is available."""
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        logger.warning(
            "xformers memory-efficient attention unavailable (%s); using default attention",
            exc,
        )
        return False
    logger.info("Enabled xformers memory-efficient attention")
    return True


def apply_quantization_optimizations(
    pipeline: Any,
    *,
    enable_model_cpu_offload: bool,
    enable_sequential_cpu_offload: bool,
    enable_vae_slicing: bool,
    enable_vae_tiling: bool,
    enable_attention_slicing: bool,
    enable_xformers: bool,
    device: str,
) -> None:
    """Apply diffusers memory/speed optimizations (accelerate offload, VAE, attention)."""
    configure_cuda_backends()
    using_offload = enable_model_cpu_offload or enable_sequential_cpu_offload

    if enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)

    if enable_vae_slicing:
        pipeline.enable_vae_slicing()
    if enable_vae_tiling:
        pipeline.enable_vae_tiling()

    if enable_attention_slicing:
        pipeline.enable_attention_slicing("auto")
    elif enable_xformers and not using_offload:
        try_enable_xformers(pipeline)
