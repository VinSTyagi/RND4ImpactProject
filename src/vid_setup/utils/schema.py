from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from uuid import UUID
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = Path(__file__).resolve().parents[2]

_SUPPORTED_PIPELINE_TYPES = frozenset({"svd", "ltx", "animatediff"})
_SUPPORTED_ANIMATEDIFF_BASES = frozenset({"sd15", "sdxl"})
_SUPPORTED_UPSCALE_TYPES = frozenset({"none", "ltx_latent"})

# Diffusers pipeline classes used by the wrapper for each backend type.
_PIPELINE_CLASS_BY_TYPE: dict[str, str] = {
    "svd": "StableVideoDiffusionPipeline",
    "ltx": "LTXImageToVideoPipeline",
    "animatediff": "AnimateDiffPipeline",
}

_ANIMATEDIFF_PIPELINE_BY_BASE: dict[str, str] = {
    "sd15": "AnimateDiffPipeline",
    "sdxl": "AnimateDiffSDXLPipeline",
}


def resolve_path(rel: str) -> Path | None:
    """Resolve config paths: ``data/...`` from repo root, else from ``src/``."""
    path = Path(rel)
    if path.is_absolute():
        resolved = path
    else:
        normalized = rel.replace("\\", "/")
        if normalized.startswith("data/"):
            resolved = _REPO_ROOT / path
        else:
            resolved = _SRC_ROOT / path
    return resolved if resolved.exists() else None


def _path_if_exists(path: Path) -> Path | None:
    return path if path.exists() else None


@dataclass
class VideoDiffuserConfig:
    """Diffusers model initialization (``from_pretrained``) settings."""

    type: str = "svd"
    model_path: str = "stabilityai/stable-video-diffusion-img2vid"
    variant: str | None = "fp16"
    revision: str | None = None
    scheduler: str = "euler"
    # AnimateDiff: motion module + SD/SDXL base checkpoint at model_path.
    motion_adapter_path: str | None = None
    base_pipeline: str | None = None
    # Optional explicit diffusers class override (e.g. LTXConditionPipeline).
    pipeline_class: str | None = None

    def normalized_type(self) -> str:
        return self.type.strip().lower()

    def resolved_pipeline_class(self) -> str:
        if self.pipeline_class:
            return self.pipeline_class.strip()
        pipeline_type = self.normalized_type()
        if pipeline_type == "animatediff":
            base = (self.base_pipeline or "sd15").strip().lower()
            return _ANIMATEDIFF_PIPELINE_BY_BASE.get(
                base, _PIPELINE_CLASS_BY_TYPE["animatediff"]
            )
        return _PIPELINE_CLASS_BY_TYPE.get(
            pipeline_type, _PIPELINE_CLASS_BY_TYPE["svd"]
        )


@dataclass
class QuantizationConfig:
    """Memory and throughput optimizations applied after pipeline load."""

    torch_dtype: str = "float16"
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_attention_slicing: bool = False
    enable_xformers: bool = True
    unet_enable_forward_chunking: bool = False


@dataclass
class GenerationConfig:
    """Default inference kwargs for the video pipeline (images supplied at runtime)."""

    num_inference_steps: int = 25
    default_guidance_scale: float = 1.0
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 1.0
    seed: int | None = None
    device: str = "cuda"
    num_frames: int = 25
    width: int | None = None
    height: int | None = None
    fps: int = 7
    # SVD micro-conditioning
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    decode_chunk_size: int = 8
    # LTX decode / conditioning
    decode_timestep: float = 0.05
    image_cond_noise_scale: float = 0.025
    # AnimateDiff
    animatediff_decode_chunk_size: int = 16


@dataclass
class InputOutputConfig:
    """Script-relative paths for scene images and generated videos."""

    script_path: str = "data/"
    input_subdir: str = "refined_images"
    image_template: str = "scene_{scene_number:02d}.png"
    raw_videos_subdir: str = "raw_videos"
    refined_videos_subdir: str = "refined_videos"
    video_template: str = "scene_{scene_number:02d}.mp4"
    save_raw: bool = True
    skip_existing: bool = True


@dataclass
class UpscaleConfig:
    """Optional post-generation video upscaling (separate diffusers pipeline)."""

    enabled: bool = False
    type: str = "none"
    model_path: str = "Lightricks/ltxv-spatial-upscaler-0.9.7"
    variant: str | None = None
    scheduler: str = "euler"
    num_inference_steps: int = 5
    tone_map_compression_ratio: float = 0.6


@dataclass
class VidSetupPipelineConfig:
    video_diffuser_config: VideoDiffuserConfig = field(
        default_factory=VideoDiffuserConfig
    )
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    io_config: InputOutputConfig = field(default_factory=InputOutputConfig)
    upscale_config: UpscaleConfig = field(default_factory=UpscaleConfig)


def upscale_active(config: VidSetupPipelineConfig) -> bool:
    upscale_cfg = config.upscale_config
    if not upscale_cfg.enabled:
        return False
    return upscale_cfg.type.strip().lower() not in {"", "none"}


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


def _io_section(data: dict[str, Any]) -> dict[str, Any]:
    section = _section(data, "io_config")
    if section:
        return section
    legacy = _section(data, "output_config")
    if not legacy:
        return {}
    migrated = dict(legacy)
    if "refined_images_path" in migrated and "input_subdir" not in migrated:
        migrated["input_subdir"] = migrated.pop("refined_images_path")
    if "filename_template" in migrated and "video_template" not in migrated:
        migrated["video_template"] = migrated.pop("filename_template")
    return migrated


def load_config(path: str) -> VidSetupPipelineConfig:
    data = _load_yaml(path)
    return VidSetupPipelineConfig(
        video_diffuser_config=VideoDiffuserConfig(
            **_dataclass_kwargs(
                VideoDiffuserConfig, _section(data, "video_diffuser_config")
            )
        ),
        quantization_config=QuantizationConfig(
            **_dataclass_kwargs(
                QuantizationConfig, _section(data, "quantization_config")
            )
        ),
        generation_config=GenerationConfig(
            **_dataclass_kwargs(GenerationConfig, _section(data, "generation_config"))
        ),
        io_config=InputOutputConfig(
            **_dataclass_kwargs(InputOutputConfig, _io_section(data))
        ),
        upscale_config=UpscaleConfig(
            **_dataclass_kwargs(UpscaleConfig, _section(data, "upscale_config"))
        ),
    )


def validate_pipeline_config(config: VidSetupPipelineConfig) -> None:
    """Reject incompatible pipeline, IO, and upscale combinations."""
    video_cfg = config.video_diffuser_config
    pipeline_type = video_cfg.normalized_type()
    if pipeline_type not in _SUPPORTED_PIPELINE_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_PIPELINE_TYPES))
        raise ValueError(
            f"unsupported video pipeline type {pipeline_type!r}; supported: {supported}"
        )

    if pipeline_type == "animatediff":
        if not video_cfg.motion_adapter_path:
            raise ValueError(
                "animatediff requires motion_adapter_path in video_diffuser_config"
            )
        base = (video_cfg.base_pipeline or "sd15").strip().lower()
        if base not in _SUPPORTED_ANIMATEDIFF_BASES:
            supported = ", ".join(sorted(_SUPPORTED_ANIMATEDIFF_BASES))
            raise ValueError(
                f"unsupported animatediff base_pipeline {base!r}; supported: {supported}"
            )

    if pipeline_type == "svd":
        gen_cfg = config.generation_config
        if gen_cfg.default_guidance_scale > 3.0:
            raise ValueError(
                "svd pipelines expect default_guidance_scale <= 3.0 "
                f"(got {gen_cfg.default_guidance_scale})"
            )

    if (
        pipeline_type == "ltx"
        and config.quantization_config.torch_dtype.lower()
        not in {
            "bfloat16",
            "bf16",
            "float16",
            "fp16",
        }
    ):
        raise ValueError("ltx pipelines require torch_dtype float16 or bfloat16")

    if not upscale_active(config):
        return

    upscale_type = config.upscale_config.type.strip().lower()
    if upscale_type not in _SUPPORTED_UPSCALE_TYPES - {"none"}:
        supported = ", ".join(sorted(_SUPPORTED_UPSCALE_TYPES - {"none"}))
        raise ValueError(
            f"unsupported upscale type {upscale_type!r}; supported: {supported}"
        )
    if upscale_type == "ltx_latent" and pipeline_type != "ltx":
        raise ValueError("ltx_latent upscaling requires video_diffuser_config.type ltx")


def _scene_image_filename(scene_number: int, io_cfg: InputOutputConfig) -> str:
    return io_cfg.image_template.format(scene_number=scene_number)


def _scene_video_filename(scene_number: int, io_cfg: InputOutputConfig) -> str:
    return io_cfg.video_template.format(scene_number=scene_number)


def scene_image_path(
    script_id: UUID | str,
    scene_number: int,
    io_cfg: InputOutputConfig,
) -> Path | None:
    base = resolve_path(io_cfg.script_path)
    if base is None:
        return None
    return _path_if_exists(
        base
        / str(script_id)
        / io_cfg.input_subdir
        / _scene_image_filename(scene_number, io_cfg)
    )


def scene_raw_video_path(
    script_id: UUID | str,
    scene_number: int,
    io_cfg: InputOutputConfig,
) -> Path | None:
    base = resolve_path(io_cfg.script_path)
    if base is None:
        return None
    return _path_if_exists(
        base
        / str(script_id)
        / io_cfg.raw_videos_subdir
        / _scene_video_filename(scene_number, io_cfg)
    )


def scene_refined_video_path(
    script_id: UUID | str,
    scene_number: int,
    io_cfg: InputOutputConfig,
) -> Path | None:
    base = resolve_path(io_cfg.script_path)
    if base is None:
        return None
    return _path_if_exists(
        base
        / str(script_id)
        / io_cfg.refined_videos_subdir
        / _scene_video_filename(scene_number, io_cfg)
    )


def _resolve_path(rel: str) -> Path:
    """Resolve config paths without requiring the path to exist."""
    path = Path(rel)
    if path.is_absolute():
        return path
    normalized = rel.replace("\\", "/")
    if normalized.startswith("data/"):
        return _REPO_ROOT / path
    return _SRC_ROOT / path


def script_base_path(io_cfg: InputOutputConfig) -> Path | None:
    base = _resolve_path(io_cfg.script_path)
    return base if base.is_dir() else None


def scene_raw_video_output_path(
    script_id: UUID | str,
    scene_number: int,
    io_cfg: InputOutputConfig,
) -> Path:
    return (
        _resolve_path(io_cfg.script_path)
        / str(script_id)
        / io_cfg.raw_videos_subdir
        / _scene_video_filename(scene_number, io_cfg)
    )


def scene_refined_video_output_path(
    script_id: UUID | str,
    scene_number: int,
    io_cfg: InputOutputConfig,
) -> Path:
    return (
        _resolve_path(io_cfg.script_path)
        / str(script_id)
        / io_cfg.refined_videos_subdir
        / _scene_video_filename(scene_number, io_cfg)
    )


def load_images(script_id: UUID | str, io_config: InputOutputConfig) -> list[Path]:
    results: list[Path] = []
    scene_number = 1
    while True:
        path = scene_image_path(script_id, scene_number, io_config)
        if path is None:
            break
        results.append(path)
        scene_number += 1
    return results


def get_uuids(io_config: InputOutputConfig) -> list[str]:
    base = script_base_path(io_config)
    if base is None:
        return []

    result: list[str] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        try:
            UUID(entry.name)
        except ValueError:
            continue
        result.append(entry.name)
    return result
