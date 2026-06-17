import contextlib
import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import Any


import torch

from diffusers import (
    CogVideoXImageToVideoPipeline,
    LTXImageToVideoPipeline,
    SanaImageToVideoPipeline,
    StableVideoDiffusionPipeline,
    WanImageToVideoPipeline,
)

from diffusers.utils import export_to_video

from PIL import Image


from utils.schema import (
    GenerationConfig,
    QuantizationConfig,
    UpscaleConfig,
    VideoDiffuserConfig,
    pipeline_generation_kwargs,
    pipeline_needs_prompt,
)

from utils.vid_diffusion_optimizations import apply_vid_optimizations


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


_PIPELINE_CLASS_BY_TYPE = {
    "svd": StableVideoDiffusionPipeline,
    "ltx": LTXImageToVideoPipeline,
    "sana": SanaImageToVideoPipeline,
    "cogvideox": CogVideoXImageToVideoPipeline,
    "wan": WanImageToVideoPipeline,
}


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


def _load_wan_pipeline(
    model_path: str,
    quant_config: QuantizationConfig,
) -> WanImageToVideoPipeline:

    from diffusers import AutoencoderKLWan

    from transformers import CLIPVisionModel

    dtype = _resolve_torch_dtype(quant_config.torch_dtype)

    image_encoder = CLIPVisionModel.from_pretrained(
        model_path,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )

    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    return WanImageToVideoPipeline.from_pretrained(
        model_path,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )


def _load_pipeline(
    init_config: VideoDiffuserConfig,
    quant_config: QuantizationConfig,
) -> Any:

    pipeline_type = init_config.normalized_type()

    if pipeline_type == "wan":
        return _load_wan_pipeline(init_config.model_path, quant_config)

    pipeline_cls = _PIPELINE_CLASS_BY_TYPE.get(pipeline_type)

    if pipeline_cls is None:
        supported = ", ".join(sorted(_PIPELINE_CLASS_BY_TYPE))

        raise ValueError(
            f"unsupported video pipeline type {pipeline_type!r}; supported: {supported}"
        )

    load_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_torch_dtype(quant_config.torch_dtype),
    }

    variant = _normalize_variant(init_config.variant)

    if variant is not None:
        load_kwargs["variant"] = variant

    return pipeline_cls.from_pretrained(init_config.model_path, **load_kwargs)


def _extract_frames(result: Any) -> list:

    if hasattr(result, "frames"):
        frames = result.frames

        if isinstance(frames, list) and frames:
            first = frames[0]

            if isinstance(first, list):
                return first

            return frames

    if isinstance(result, list):
        return result

    raise ValueError("pipeline output did not contain video frames")


@contextlib.contextmanager
def start_vid_diff_engine(
    logger: logging.Logger,
    init_config: VideoDiffuserConfig,
    quant_config: QuantizationConfig,
):
    """Starts the video diffuser engine."""

    pipe = None

    using_offload = False

    pipeline_type = init_config.normalized_type()

    try:
        logger.info(
            "Starting video diffuser engine: type=%s model=%s device=%s torch_dtype=%s",
            pipeline_type,
            init_config.model_path,
            init_config.device,
            quant_config.torch_dtype,
        )

        pipe = _load_pipeline(init_config, quant_config)

        using_offload = apply_vid_optimizations(
            pipe,
            pipeline_type,
            quant_config,
            init_config.device,
        )

        logger.info(
            "Video diffuser engine started (type=%s offload=%s)",
            pipeline_type,
            using_offload,
        )

        yield pipe, using_offload

    except Exception as exc:
        logger.error("Error starting video diffuser engine: %s", exc)

        raise ValueError(f"Error starting video diffuser engine: {exc}") from exc

    finally:
        if pipe is not None:
            del pipe

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Video diffuser engine stopped")


def start_upscale_engine(init_config: UpscaleConfig):

    pass


def generate_video_from_images(
    logger: logging.Logger,
    pipe: Any,
    gen_config: GenerationConfig,
    pipeline_type: str,
    input_path: str,
    output_path: str,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    using_offload: bool = False,
):
    """Generates a video from a given input path and output path."""

    try:
        image = Image.open(input_path).convert("RGB")

    except Exception as exc:
        raise ValueError(f"Error opening image: {exc}") from exc

    image = image.resize(
        (gen_config.width, gen_config.height), Image.Resampling.LANCZOS
    )

    try:
        args = pipeline_generation_kwargs(pipeline_type, gen_config)

        normalized_type = pipeline_type.strip().lower()

        if pipeline_needs_prompt(normalized_type):
            resolved_prompt = (prompt or gen_config.prompt or "").strip()

            resolved_negative = (
                negative_prompt or gen_config.negative_prompt or ""
            ).strip()

            if not resolved_prompt:
                raise ValueError(
                    f"{normalized_type} generation requires prompt "
                    "(set generation_config.prompt or populate image_prompt in script.json)"
                )

            args["prompt"] = resolved_prompt

            if resolved_negative:
                args["negative_prompt"] = resolved_negative

            args["width"] = gen_config.width

            args["height"] = gen_config.height

        result = pipe(image=image, **args)

        frames = _extract_frames(result)

        export_fps = (
            gen_config.fps if gen_config.fps is not None else gen_config.frame_rate
        )

        if export_fps is None:
            raise ValueError(
                "generation_config must set fps (svd) or frame_rate (prompted backends) "
                "for export"
            )
        _export_video_atomically(frames, output_path, fps=export_fps)
        if using_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output_path
    except Exception as exc:
        raise ValueError(f"Error generating video: {exc}") from exc


def _export_video_atomically(frames: list, output_path: str, *, fps: int) -> None:
    """Write each scene video via a temp file so partial outputs are not left behind."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=destination.suffix or ".mp4",
            dir=destination.parent,
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        export_to_video(frames, str(tmp_path), fps=fps)
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def upscale_video(
    pipe: Any,
    gen_config: GenerationConfig,
    input_path: str,
    output_path: str,
):

    pass
