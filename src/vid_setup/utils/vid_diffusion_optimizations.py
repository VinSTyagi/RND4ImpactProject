from __future__ import annotations

import logging
from typing import Any

import torch

from utils.schema import QuantizationConfig

logger = logging.getLogger(__name__)

_CUDA_BACKENDS_CONFIGURED = False
# CogVideoX passes image_rotary_emb via cross_attention_kwargs; XFormersAttnProcessor ignores it.
_XFORMERS_INCOMPATIBLE_PIPELINE_TYPES = frozenset({"cogvideox"})


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
    if not hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        logger.debug("Pipeline does not support xformers memory-efficient attention")
        return False
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


def _try_enable_vae_slicing(pipeline: Any) -> None:
    if hasattr(pipeline, "enable_vae_slicing"):
        try:
            pipeline.enable_vae_slicing()
            logger.info("Enabled VAE slicing")
            return
        except NotImplementedError as exc:
            logger.debug("Pipeline VAE slicing unavailable: %s", exc)
        except Exception as exc:
            logger.warning("Pipeline VAE slicing failed (%s); skipping", exc)
    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "enable_slicing"):
        try:
            vae.enable_slicing()
            logger.info("Enabled VAE slicing on pipeline.vae")
            return
        except NotImplementedError as exc:
            logger.debug(
                "VAE slicing not implemented for %s: %s", type(vae).__name__, exc
            )
        except Exception as exc:
            logger.warning("VAE slicing on pipeline.vae failed (%s); skipping", exc)
    logger.debug("VAE slicing not supported by this pipeline")


def _try_enable_vae_tiling(pipeline: Any) -> None:
    if hasattr(pipeline, "enable_vae_tiling"):
        try:
            pipeline.enable_vae_tiling()
            logger.info("Enabled VAE tiling")
            return
        except NotImplementedError as exc:
            logger.debug("Pipeline VAE tiling unavailable: %s", exc)
        except Exception as exc:
            logger.warning("Pipeline VAE tiling failed (%s); skipping", exc)
    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
            logger.info("Enabled VAE tiling on pipeline.vae")
            return
        except NotImplementedError as exc:
            logger.debug(
                "VAE tiling not implemented for %s: %s", type(vae).__name__, exc
            )
        except Exception as exc:
            logger.warning("VAE tiling on pipeline.vae failed (%s); skipping", exc)
    logger.debug("VAE tiling not supported by this pipeline")


def _try_enable_attention_slicing(pipeline: Any) -> None:
    if not hasattr(pipeline, "enable_attention_slicing"):
        logger.debug("Attention slicing not supported by this pipeline")
        return
    try:
        pipeline.enable_attention_slicing("auto")
        logger.info("Enabled attention slicing")
    except Exception as exc:
        logger.warning("Attention slicing unavailable (%s); skipping", exc)


def _apply_svd_offload(
    pipeline: Any,
    quant_config: QuantizationConfig,
    device: str,
) -> bool:
    using_offload = (
        quant_config.enable_model_cpu_offload
        or quant_config.enable_sequential_cpu_offload
    )
    if quant_config.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif quant_config.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif device and device != "cpu":
        pipeline.to(device)
    return using_offload


def _apply_ltx_group_offload(pipeline: Any, device: str) -> bool:
    """Apply diffusers group offloading for LTX pipelines (recommended for low VRAM)."""
    try:
        from diffusers.hooks import apply_group_offloading
    except ImportError as exc:
        logger.warning(
            "group offloading unavailable (%s); falling back to sequential CPU offload",
            exc,
        )
        pipeline.enable_sequential_cpu_offload()
        return True

    onload_device = torch.device(device if device and device != "cpu" else "cuda")
    offload_device = torch.device("cpu")

    transformer = getattr(pipeline, "transformer", None)
    if transformer is not None and hasattr(transformer, "enable_group_offload"):
        transformer.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
        )

    text_encoder = getattr(pipeline, "text_encoder", None)
    if text_encoder is not None:
        apply_group_offloading(
            text_encoder,
            onload_device=onload_device,
            offload_type="block_level",
            num_blocks_per_group=2,
        )

    vae = getattr(pipeline, "vae", None)
    if vae is not None:
        apply_group_offloading(
            vae,
            onload_device=onload_device,
            offload_type="leaf_level",
        )

    logger.info("Enabled LTX group offloading (onload=%s)", onload_device)
    return True


def _apply_ltx_offload(
    pipeline: Any,
    quant_config: QuantizationConfig,
    device: str,
) -> bool:
    if quant_config.enable_group_offload:
        return _apply_ltx_group_offload(pipeline, device)

    using_offload = (
        quant_config.enable_model_cpu_offload
        or quant_config.enable_sequential_cpu_offload
    )
    if quant_config.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif quant_config.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif device and device != "cpu":
        pipeline.to(device)
    return using_offload


def apply_vid_optimizations(
    pipeline: Any,
    pipeline_type: str,
    quant_config: QuantizationConfig,
    device: str,
) -> bool:
    """Apply backend-specific memory optimizations. Returns True when offloading is active."""
    configure_cuda_backends()
    normalized = pipeline_type.strip().lower()

    if normalized == "svd":
        using_offload = _apply_svd_offload(pipeline, quant_config, device)
        if quant_config.unet_enable_forward_chunking:
            unet = getattr(pipeline, "unet", None)
            if unet is not None and hasattr(unet, "enable_forward_chunking"):
                unet.enable_forward_chunking()
                logger.info("Enabled SVD UNet forward chunking")
    elif normalized == "ltx":
        using_offload = _apply_ltx_offload(pipeline, quant_config, device)
    else:
        using_offload = _apply_svd_offload(pipeline, quant_config, device)

    if quant_config.enable_vae_slicing:
        _try_enable_vae_slicing(pipeline)
    if quant_config.enable_vae_tiling:
        _try_enable_vae_tiling(pipeline)

    if quant_config.enable_attention_slicing:
        _try_enable_attention_slicing(pipeline)
    elif quant_config.enable_xformers and not using_offload:
        if normalized in _XFORMERS_INCOMPATIBLE_PIPELINE_TYPES:
            logger.info(
                "Skipping xformers for %s (requires image_rotary_emb not supported by XFormersAttnProcessor)",
                normalized,
            )
        else:
            try_enable_xformers(pipeline)

    return using_offload
