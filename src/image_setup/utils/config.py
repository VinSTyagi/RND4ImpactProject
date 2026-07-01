from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from utils.schema import ImagePrompt, resolve_path


_ASPECT_SIZES_SDXL: dict[str, tuple[int, int]] = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (1024, 1024),
}

_ASPECT_SIZES_SD15: dict[str, tuple[int, int]] = {
    "16:9": (768, 432),
    "9:16": (432, 768),
    "1:1": (512, 512),
}

_ASPECT_SIZES_BY_PIPELINE: dict[str, dict[str, tuple[int, int]]] = {
    "sdxl": _ASPECT_SIZES_SDXL,
    "sd15": _ASPECT_SIZES_SD15,
}

_SUPPORTED_PIPELINE_TYPES = frozenset(_ASPECT_SIZES_BY_PIPELINE)


@dataclass
class DiffusionPipelineConfig:
    type: str = "sdxl"
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    variant: str | None = "fp16"
    scheduler: str = "euler"
    unet_checkpoint_repo: str | None = None
    unet_checkpoint_file: str | None = None

    def uses_distilled_low_cfg(self) -> bool:
        if "turbo" in self.model_path.lower():
            return True
        repo = self.unet_checkpoint_repo or ""
        return "lightning" in repo.lower()

    def uses_lightning_unet(self) -> bool:
        return bool(self.unet_checkpoint_repo and self.unet_checkpoint_file)


@dataclass
class QuantizationConfig:
    torch_dtype: str = "float16"
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_attention_slicing: bool = False
    enable_xformers: bool = True


@dataclass
class GenerationConfig:
    num_inference_steps: int = 30
    default_guidance_scale: float = 7.5
    seed: int | None = None
    device: str = "cuda"
    width: int | None = None
    height: int | None = None


@dataclass
class RefinementConfig:
    enabled: bool = False
    type: str = "none"
    model_path: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    variant: str | None = "fp16"
    scheduler: str = "euler"
    num_inference_steps: int = 40
    denoising_start: float = 0.8
    denoising_end: float = 0.8
    image_denoising_start: float = 0.01
    strength: float = 0.35


@dataclass
class OutputConfig:
    script_path: str = "data/"
    raw_subdir: str = "raw_images"
    output_subdir: str = "raw_images"
    filename_template: str = "scene_{scene_number:02d}_{prompt_number}.png"
    save_raw: bool = True
    skip_existing: bool = True


@dataclass
class ImageSetupPipelineConfig:
    pipeline_config: DiffusionPipelineConfig = field(
        default_factory=DiffusionPipelineConfig
    )
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    refinement_config: RefinementConfig = field(default_factory=RefinementConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)


def refinement_active(config: ImageSetupPipelineConfig) -> bool:
    ref_cfg = config.refinement_config
    if not ref_cfg.enabled:
        return False
    return ref_cfg.type.strip().lower() not in {"", "none"}


def _load_yaml(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return _simple_yaml_parse(text)


def _simple_yaml_parse(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1].strip()
            out[key] = {}
            current = out[key]
            continue
        if current is None or ":" not in line:
            continue
        k, v = line.strip().split(":", 1)
        current[k.strip()] = _parse_scalar(v.strip())
    return out


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _dataclass_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    valid = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in valid}


def load_config(path: str) -> ImageSetupPipelineConfig:
    data = _load_yaml(path)
    config = ImageSetupPipelineConfig(
        pipeline_config=DiffusionPipelineConfig(
            **_dataclass_kwargs(DiffusionPipelineConfig, _section(data, "pipeline_config"))
        ),
        quantization_config=QuantizationConfig(
            **_dataclass_kwargs(QuantizationConfig, _section(data, "quantization_config"))
        ),
        generation_config=GenerationConfig(
            **_dataclass_kwargs(GenerationConfig, _section(data, "generation_config"))
        ),
        refinement_config=RefinementConfig(
            **_dataclass_kwargs(RefinementConfig, _section(data, "refinement_config"))
        ),
        output_config=OutputConfig(
            **_dataclass_kwargs(OutputConfig, _section(data, "output_config"))
        ),
    )
    validate_pipeline_config(config)
    return config


def parse_cfg_scale(value: str, default: float) -> float:
    raw = str(value).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid cfg_scale: {value!r}") from exc


def validate_pipeline_config(config: ImageSetupPipelineConfig) -> None:
    pipeline_cfg = config.pipeline_config
    pipeline_type = pipeline_cfg.type.strip().lower()
    if pipeline_type not in _SUPPORTED_PIPELINE_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_PIPELINE_TYPES))
        raise ValueError(
            f"unsupported pipeline type {pipeline_type!r}; supported: {supported}"
        )

    repo = pipeline_cfg.unet_checkpoint_repo
    ckpt_file = pipeline_cfg.unet_checkpoint_file
    if (repo is None) ^ (ckpt_file is None):
        raise ValueError(
            "unet_checkpoint_repo and unet_checkpoint_file must both be set or omitted"
        )
    if pipeline_cfg.uses_lightning_unet() and pipeline_type != "sdxl":
        raise ValueError("lightning UNet checkpoints require pipeline type sdxl")

    if not refinement_active(config):
        width = config.generation_config.width
        height = config.generation_config.height
        if (width is None) ^ (height is None):
            raise ValueError(
                "generation_config width and height must both be set or both omitted"
            )
        if width is not None and height is not None:
            _validate_resolution(width, height)
        return

    ref_cfg = config.refinement_config
    ref_type = ref_cfg.type.strip().lower()
    if pipeline_type == "sd15" and ref_type == "sdxl_refiner":
        raise ValueError(
            "sd15 pipelines do not support sdxl_refiner refinement; "
            "use img2img or disable refinement"
        )
    if pipeline_cfg.uses_lightning_unet() and ref_type == "sdxl_refiner":
        raise ValueError(
            "lightning UNet checkpoints do not support sdxl_refiner refinement; "
            "use img2img or disable refinement"
        )
    if ref_type == "sdxl_refiner":
        if not (0.0 < ref_cfg.denoising_start < 1.0):
            raise ValueError(
                f"refinement_config.denoising_start must be in (0, 1), "
                f"got {ref_cfg.denoising_start}"
            )
        if ref_cfg.denoising_start != ref_cfg.denoising_end:
            raise ValueError(
                "refinement_config.denoising_start and denoising_end must match "
                f"for SDXL latent handoff ({ref_cfg.denoising_start} != "
                f"{ref_cfg.denoising_end})"
            )
        if not (0.0 <= ref_cfg.image_denoising_start < ref_cfg.denoising_start):
            raise ValueError(
                "refinement_config.image_denoising_start must be in "
                f"[0, denoising_start) for PNG refine-from-disk; got "
                f"{ref_cfg.image_denoising_start}"
            )

    width = config.generation_config.width
    height = config.generation_config.height
    if (width is None) ^ (height is None):
        raise ValueError(
            "generation_config width and height must both be set or both omitted"
        )
    if width is not None and height is not None:
        _validate_resolution(width, height)


def resolve_aspect_size(
    aspect_ratio: str,
    pipeline_type: str = "sdxl",
) -> tuple[int, int]:
    normalized_type = str(pipeline_type).strip().lower()
    sizes = _ASPECT_SIZES_BY_PIPELINE.get(normalized_type)
    if sizes is None:
        supported = ", ".join(sorted(_SUPPORTED_PIPELINE_TYPES))
        raise ValueError(
            f"unsupported pipeline type {pipeline_type!r}; supported: {supported}"
        )
    normalized = str(aspect_ratio).strip()
    if normalized not in sizes:
        valid = ", ".join(sorted(sizes))
        raise ValueError(
            f"invalid aspect_ratio {aspect_ratio!r} for {normalized_type}; "
            f"expected one of: {valid}"
        )
    return sizes[normalized]


def _validate_resolution(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(
            f"width and height must be multiples of 8, got {width}x{height}"
        )


def resolve_generation_size(
    image_prompt: ImagePrompt,
    pipeline_type: str,
    gen_cfg: GenerationConfig,
) -> tuple[int, int]:
    if gen_cfg.width is None and gen_cfg.height is None:
        return resolve_aspect_size(image_prompt["aspect_ratio"], pipeline_type)
    if gen_cfg.width is None or gen_cfg.height is None:
        raise ValueError(
            "generation_config width and height must both be set when overriding "
            "scene aspect_ratio"
        )
    return int(gen_cfg.width), int(gen_cfg.height)


def format_positive_prompt(tags: list[str]) -> str:
    from utils.image_prompt import join_prompt_tags, truncate_tags_to_clip

    return join_prompt_tags(truncate_tags_to_clip(tags))


def format_negative_prompt(tags: list[str]) -> str:
    from utils.image_prompt import join_prompt_tags, truncate_tags_to_clip

    return join_prompt_tags(truncate_tags_to_clip(tags))


def _scene_filename(
    scene_number: int,
    prompt_number: int,
    output_cfg: OutputConfig,
) -> str:
    return output_cfg.filename_template.format(
        scene_number=scene_number,
        prompt_number=prompt_number,
    )


def scene_prompt_output_path(
    script_id: UUID | str,
    scene_number: int,
    prompt_number: int,
    output_cfg: OutputConfig,
    *,
    subdir: str,
) -> Path:
    return (
        resolve_path(output_cfg.script_path)
        / str(script_id)
        / str(scene_number)
        / subdir
        / _scene_filename(scene_number, prompt_number, output_cfg)
    )


def scene_output_path(
    script_id: UUID | str,
    scene_number: int,
    prompt_number: int,
    output_cfg: OutputConfig,
) -> Path:
    return scene_prompt_output_path(
        script_id,
        scene_number,
        prompt_number,
        output_cfg,
        subdir=output_cfg.output_subdir,
    )


def scene_raw_output_path(
    script_id: UUID | str,
    scene_number: int,
    prompt_number: int,
    output_cfg: OutputConfig,
) -> Path:
    return scene_prompt_output_path(
        script_id,
        scene_number,
        prompt_number,
        output_cfg,
        subdir=output_cfg.raw_subdir,
    )
