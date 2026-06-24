from __future__ import annotations

import contextlib
import gc
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from utils.diffusion_optimizations import apply_quantization_optimizations
from utils.schema import (
    format_negative_prompt,
    format_positive_prompt,
    parse_cfg_scale,
    resolve_generation_size,
)

if TYPE_CHECKING:
    from utils.schema import (
        DiffusionPipelineConfig,
        GenerationConfig,
        ImagePrompt,
        QuantizationConfig,
        RefinementConfig,
    )

logger = logging.getLogger(__name__)

PipelineType = Literal["sdxl", "sd15"]
OutputType = Literal["pil", "latent"]

_TURBO_CFG_MAX = 2.0
_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class DiffusionTxt2ImgPipeline(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


class DiffusionImg2ImgPipeline(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class PipelineFamily:
    name: PipelineType
    txt2img_cls: type
    img2img_cls: type
    supports_refiner: bool
    supports_latent_handoff: bool


_PIPELINE_FAMILIES: dict[PipelineType, PipelineFamily] = {
    "sdxl": PipelineFamily(
        name="sdxl",
        txt2img_cls=StableDiffusionXLPipeline,
        img2img_cls=StableDiffusionXLImg2ImgPipeline,
        supports_refiner=True,
        supports_latent_handoff=True,
    ),
    "sd15": PipelineFamily(
        name="sd15",
        txt2img_cls=StableDiffusionPipeline,
        img2img_cls=StableDiffusionImg2ImgPipeline,
        supports_refiner=False,
        supports_latent_handoff=False,
    ),
}


def normalize_pipeline_type(pipeline_type: str) -> PipelineType:
    normalized = str(pipeline_type).strip().lower()
    if normalized not in _PIPELINE_FAMILIES:
        supported = ", ".join(sorted(_PIPELINE_FAMILIES))
        raise ValueError(
            f"unsupported pipeline type {pipeline_type!r}; supported: {supported}"
        )
    return normalized  # type: ignore[return-value]


def get_pipeline_family(pipeline_type: str) -> PipelineFamily:
    return _PIPELINE_FAMILIES[normalize_pipeline_type(pipeline_type)]


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    key = str(dtype_name).strip().lower()
    if key not in _DTYPE_MAP:
        valid = ", ".join(sorted(_DTYPE_MAP))
        raise ValueError(
            f"unsupported torch_dtype {dtype_name!r}; expected one of: {valid}"
        )
    return _DTYPE_MAP[key]


def _normalize_variant(variant: str | None) -> str | None:
    if variant is not None and str(variant).strip().lower() in {"", "null", "none"}:
        return None
    return variant


def _build_load_kwargs(
    pipeline_cfg: DiffusionPipelineConfig,
    quant_cfg: QuantizationConfig,
    *,
    variant: str | None = None,
) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_torch_dtype(quant_cfg.torch_dtype),
        "use_safetensors": True,
    }
    resolved_variant = _normalize_variant(
        pipeline_cfg.variant if variant is None else variant
    )
    if resolved_variant:
        load_kwargs["variant"] = resolved_variant
    return load_kwargs


def _apply_scheduler(
    pipeline: Any,
    scheduler_name: str,
    *,
    lightning_unet: bool = False,
) -> None:
    if lightning_unet:
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing",
        )
        return

    normalized = str(scheduler_name).strip().lower()
    if normalized in {"", "default"}:
        return
    if normalized == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
        return
    if normalized in {"dpm++", "dpm", "dpm_solver"}:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        return
    raise ValueError(
        f"unsupported scheduler {scheduler_name!r}; expected euler, dpm++, or default"
    )


def _load_lightning_unet(
    pipeline: Any,
    pipeline_cfg: DiffusionPipelineConfig,
) -> None:
    repo = pipeline_cfg.unet_checkpoint_repo
    filename = pipeline_cfg.unet_checkpoint_file
    if not repo or not filename:
        return

    logger.info("Loading lightning UNet checkpoint %s/%s", repo, filename)
    checkpoint_path = hf_hub_download(repo, filename)
    state_dict = load_file(checkpoint_path)
    pipeline.unet.load_state_dict(state_dict)


def _prepare_loaded_pipeline(
    pipeline: Any,
    pipeline_cfg: DiffusionPipelineConfig,
    quant_cfg: QuantizationConfig,
    gen_cfg: GenerationConfig,
    *,
    scheduler_name: str | None = None,
) -> None:
    _load_lightning_unet(pipeline, pipeline_cfg)
    _apply_scheduler(
        pipeline,
        scheduler_name if scheduler_name is not None else pipeline_cfg.scheduler,
        lightning_unet=pipeline_cfg.uses_lightning_unet(),
    )
    _apply_quantization(pipeline, quant_cfg, gen_cfg)


def _apply_quantization(
    pipeline: Any,
    quant_cfg: QuantizationConfig,
    gen_cfg: GenerationConfig,
) -> None:
    apply_quantization_optimizations(
        pipeline,
        enable_model_cpu_offload=quant_cfg.enable_model_cpu_offload,
        enable_sequential_cpu_offload=quant_cfg.enable_sequential_cpu_offload,
        enable_vae_slicing=quant_cfg.enable_vae_slicing,
        enable_vae_tiling=quant_cfg.enable_vae_tiling,
        enable_attention_slicing=quant_cfg.enable_attention_slicing,
        enable_xformers=quant_cfg.enable_xformers,
        device=gen_cfg.device,
    )


def load_txt2img_pipeline(
    pipeline_cfg: DiffusionPipelineConfig,
    quant_cfg: QuantizationConfig,
    gen_cfg: GenerationConfig,
) -> DiffusionTxt2ImgPipeline:
    family = get_pipeline_family(pipeline_cfg.type)
    load_kwargs = _build_load_kwargs(pipeline_cfg, quant_cfg)
    logger.info(
        "Loading %s txt2img pipeline from %s",
        family.name,
        pipeline_cfg.model_path,
    )
    pipeline = family.txt2img_cls.from_pretrained(
        pipeline_cfg.model_path,
        **load_kwargs,
    )
    _prepare_loaded_pipeline(pipeline, pipeline_cfg, quant_cfg, gen_cfg)
    logger.info(
        "%s txt2img pipeline ready (device=%s, offload=%s, scheduler=%s)",
        family.name,
        gen_cfg.device,
        quant_cfg.enable_model_cpu_offload or quant_cfg.enable_sequential_cpu_offload,
        pipeline_cfg.scheduler,
    )
    return pipeline


def load_img2img_pipeline(
    model_path: str,
    pipeline_cfg: DiffusionPipelineConfig,
    quant_cfg: QuantizationConfig,
    gen_cfg: GenerationConfig,
    *,
    scheduler: str | None = None,
    base_pipeline: DiffusionTxt2ImgPipeline | None = None,
    variant: str | None = None,
) -> DiffusionImg2ImgPipeline:
    family = get_pipeline_family(pipeline_cfg.type)
    load_kwargs = _build_load_kwargs(pipeline_cfg, quant_cfg, variant=variant)
    scheduler_name = scheduler if scheduler is not None else pipeline_cfg.scheduler

    if base_pipeline is not None and family.name == "sdxl":
        logger.info(
            "Loading %s img2img pipeline from %s (shared encoders)",
            family.name,
            model_path,
        )
        pipeline = family.img2img_cls.from_pretrained(
            model_path,
            text_encoder_2=base_pipeline.text_encoder_2,
            vae=base_pipeline.vae,
            **load_kwargs,
        )
    else:
        logger.info("Loading %s img2img pipeline from %s", family.name, model_path)
        pipeline = family.img2img_cls.from_pretrained(
            model_path,
            **load_kwargs,
        )

    _prepare_loaded_pipeline(
        pipeline,
        pipeline_cfg,
        quant_cfg,
        gen_cfg,
        scheduler_name=scheduler_name,
    )
    logger.info(
        "%s img2img pipeline ready (device=%s, scheduler=%s)",
        family.name,
        gen_cfg.device,
        scheduler_name,
    )
    return pipeline


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_guidance_scale(
    image_prompt: ImagePrompt,
    gen_cfg: GenerationConfig,
    pipeline_cfg: DiffusionPipelineConfig,
) -> float:
    guidance_scale = parse_cfg_scale(
        image_prompt["cfg_scale"],
        gen_cfg.default_guidance_scale,
    )
    if pipeline_cfg.uses_distilled_low_cfg():
        guidance_scale = min(max(guidance_scale, 0.0), _TURBO_CFG_MAX)
    return guidance_scale


def _resolve_generator(device: str, gen_cfg: GenerationConfig, seed_offset: int):
    if gen_cfg.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = int(gen_cfg.seed) + int(seed_offset)
    generator_device = "cpu" if device == "cpu" else device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator, seed


def _prompt_bundle(
    image_prompt: ImagePrompt,
    pipeline_type: str,
    gen_cfg: GenerationConfig,
) -> tuple[str, str, int, int]:
    positive = format_positive_prompt(image_prompt["positive_prompt"])
    negative = format_negative_prompt(image_prompt["negative_prompt"])
    width, height = resolve_generation_size(image_prompt, pipeline_type, gen_cfg)
    return positive, negative, width, height


def generate_scene_image(
    pipeline: DiffusionTxt2ImgPipeline,
    image_prompt: ImagePrompt,
    gen_cfg: GenerationConfig,
    *,
    pipeline_type: str,
    pipeline_cfg: DiffusionPipelineConfig,
    seed_offset: int = 0,
    output_type: OutputType = "pil",
    denoising_end: float | None = None,
) -> Image.Image | torch.Tensor:
    family = get_pipeline_family(pipeline_type)
    positive, negative, width, height = _prompt_bundle(
        image_prompt, pipeline_type, gen_cfg
    )
    guidance_scale = _resolve_guidance_scale(image_prompt, gen_cfg, pipeline_cfg)
    generator, seed = _resolve_generator(gen_cfg.device, gen_cfg, seed_offset)

    pipeline_output_type = "latent" if output_type == "latent" else "pil"
    call_kwargs: dict[str, Any] = {
        "prompt": positive,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "num_inference_steps": gen_cfg.num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "output_type": pipeline_output_type,
    }
    if denoising_end is not None:
        if not family.supports_latent_handoff:
            raise ValueError(
                f"{family.name} pipelines do not support latent handoff (denoising_end)"
            )
        call_kwargs["denoising_end"] = denoising_end

    logger.info(
        "Generating raw image (%dx%d, steps=%s, cfg=%.2f, seed=%s, output=%s, family=%s)",
        width,
        height,
        gen_cfg.num_inference_steps,
        guidance_scale,
        seed,
        pipeline_output_type,
        family.name,
    )

    result = pipeline(**call_kwargs)
    if output_type == "latent":
        latents = result.images
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                f"expected latent tensor from pipeline, got {type(latents).__name__}"
            )
        return latents
    return result.images[0]


def resolve_refiner_denoising_start(
    image: Image.Image | torch.Tensor,
    ref_cfg: RefinementConfig,
) -> float:
    """Latent tensors use the handoff point; decoded PNGs need a near-zero start."""
    if isinstance(image, Image.Image):
        return ref_cfg.image_denoising_start
    return ref_cfg.denoising_start


def refine_scene_sdxl_refiner(
    pipeline: DiffusionImg2ImgPipeline,
    image_prompt: ImagePrompt,
    gen_cfg: GenerationConfig,
    ref_cfg: RefinementConfig,
    pipeline_cfg: DiffusionPipelineConfig,
    image: Image.Image | torch.Tensor,
    seed_offset: int = 0,
) -> Image.Image:
    positive, negative, _, _ = _prompt_bundle(image_prompt, "sdxl", gen_cfg)
    guidance_scale = _resolve_guidance_scale(image_prompt, gen_cfg, pipeline_cfg)
    generator, seed = _resolve_generator(gen_cfg.device, gen_cfg, seed_offset)
    denoising_start = resolve_refiner_denoising_start(image, ref_cfg)
    input_kind = "latent" if not isinstance(image, Image.Image) else "pil"

    logger.info(
        "Refining image (input=%s, steps=%s, cfg=%.2f, denoising_start=%.3f, seed=%s)",
        input_kind,
        ref_cfg.num_inference_steps,
        guidance_scale,
        denoising_start,
        seed,
    )

    result = pipeline(
        prompt=positive,
        negative_prompt=negative,
        image=image,
        num_inference_steps=ref_cfg.num_inference_steps,
        denoising_start=denoising_start,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


def refine_scene_img2img(
    pipeline: DiffusionImg2ImgPipeline,
    image_prompt: ImagePrompt,
    gen_cfg: GenerationConfig,
    ref_cfg: RefinementConfig,
    *,
    pipeline_type: str,
    pipeline_cfg: DiffusionPipelineConfig,
    image: Image.Image,
    seed_offset: int = 0,
) -> Image.Image:
    positive, negative, _, _ = _prompt_bundle(image_prompt, pipeline_type, gen_cfg)
    guidance_scale = _resolve_guidance_scale(image_prompt, gen_cfg, pipeline_cfg)
    generator, seed = _resolve_generator(gen_cfg.device, gen_cfg, seed_offset)

    logger.info(
        "Refining image via img2img (steps=%s, cfg=%.2f, strength=%.2f, seed=%s, family=%s)",
        ref_cfg.num_inference_steps,
        guidance_scale,
        ref_cfg.strength,
        seed,
        pipeline_type,
    )

    result = pipeline(
        prompt=positive,
        negative_prompt=negative,
        image=image,
        strength=ref_cfg.strength,
        num_inference_steps=ref_cfg.num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


@contextlib.contextmanager
def txt2img_session(
    pipeline_cfg: DiffusionPipelineConfig,
    quant_cfg: QuantizationConfig,
    gen_cfg: GenerationConfig,
):
    """Load a txt2img pipeline for one run and guarantee cleanup on exit."""
    pipeline = load_txt2img_pipeline(pipeline_cfg, quant_cfg, gen_cfg)
    try:
        yield pipeline
    finally:
        del pipeline
        cleanup()


@contextlib.contextmanager
def img2img_session(
    model_path: str,
    pipeline_cfg: DiffusionPipelineConfig,
    quant_cfg: QuantizationConfig,
    gen_cfg: GenerationConfig,
    *,
    scheduler: str | None = None,
    base_pipeline: DiffusionTxt2ImgPipeline | None = None,
    variant: str | None = None,
):
    """Load an img2img pipeline for one run and guarantee cleanup on exit."""
    pipeline = load_img2img_pipeline(
        model_path,
        pipeline_cfg,
        quant_cfg,
        gen_cfg,
        scheduler=scheduler,
        base_pipeline=base_pipeline,
        variant=variant,
    )
    try:
        yield pipeline
    finally:
        del pipeline
        cleanup()


# Backward-compatible aliases for SDXL-specific call sites.
sdxl_session = txt2img_session
sdxl_img2img_session = img2img_session
load_sdxl_pipeline = load_txt2img_pipeline
load_sdxl_img2img_pipeline = load_img2img_pipeline
