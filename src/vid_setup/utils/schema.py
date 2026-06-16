from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from uuid import UUID
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = Path(__file__).resolve().parents[2]

_SUPPORTED_PIPELINE_TYPES = frozenset({"svd", "ltx"})
_SUPPORTED_UPSCALE_TYPES = frozenset({"none", "ltx_latent"})

# Diffusers pipeline classes used by the wrapper for each backend type.
_PIPELINE_CLASS_BY_TYPE: dict[str, str] = {
    "svd": "StableVideoDiffusionPipeline",
    "ltx": "LTXImageToVideoPipeline",
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


_SVD_GENERATION_FIELDS = frozenset({
    "min_guidance_scale",
    "max_guidance_scale",
    "fps",
    "motion_bucket_id",
    "noise_aug_strength",
    "decode_chunk_size",
})
_LTX_GENERATION_FIELDS = frozenset({
    "guidance_scale",
    "frame_rate",
    "decode_timestep",
    "decode_noise_scale",
})


@dataclass
class VideoDiffuserConfig:
    """Diffusers model initialization (``from_pretrained``) settings."""
    device: str
    type: str = "svd"
    model_path: str = "stabilityai/stable-video-diffusion-img2vid"
    torch_dtype: str = "float16"
    variant: str = "fp16"

    def normalized_type(self) -> str:
        return self.type.strip().lower()


@dataclass
class GenerationConfig:
    num_inference_steps: int
    num_frames: int
    width: int
    height: int

    # SVD-specific (optional)
    min_guidance_scale: float | None = None
    max_guidance_scale: float | None = None
    fps: int | None = None
    motion_bucket_id: int | None = None
    noise_aug_strength: float | None = None
    decode_chunk_size: int | None = None

    # LTX-specific (optional)
    guidance_scale: float | None = None
    frame_rate: int | None = None
    decode_timestep: float | None = None
    decode_noise_scale: float | None = None


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
    video_diffuser_config: VideoDiffuserConfig
    generation_config: GenerationConfig
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
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
    config = VidSetupPipelineConfig(
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
    validate_pipeline_config(config)
    return config


def validate_generation_config(pipeline_type: str, gen_cfg: GenerationConfig) -> None:
    """Reject pipeline-specific generation fields on the wrong backend."""
    if pipeline_type == "svd":
        if (
            gen_cfg.max_guidance_scale is not None
            and gen_cfg.max_guidance_scale > 3.0
        ):
            raise ValueError(
                "svd pipelines expect max_guidance_scale <= 3.0 "
                f"(got {gen_cfg.max_guidance_scale})"
            )
        for name in _LTX_GENERATION_FIELDS:
            if getattr(gen_cfg, name) is not None:
                raise ValueError(
                    f"svd generation_config must not set ltx-only field {name!r}"
                )
        return

    if pipeline_type == "ltx":
        for name in _SVD_GENERATION_FIELDS:
            if getattr(gen_cfg, name) is not None:
                raise ValueError(
                    f"ltx generation_config must not set svd-only field {name!r}"
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

    validate_generation_config(pipeline_type, config.generation_config)

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


def _resolve_path(rel: str) -> Path:
    """Resolve config paths without requiring the path to exist."""
    path = Path(rel)
    if path.is_absolute():
        return path
    normalized = rel.replace("\\", "/")
    if normalized.startswith("data/"):
        return _REPO_ROOT / path
    return _SRC_ROOT / path


@dataclass(frozen=True)
class ScenePaths:
    image: Path
    raw_video: Path
    refined_video: Path


def scene_paths(
    script_id: UUID | str,
    scene_number: int,
    io_cfg: InputOutputConfig,
) -> ScenePaths:
    """Resolved image, raw-video, and refined-video paths for one scene."""
    base = _resolve_path(io_cfg.script_path) / str(script_id)
    return ScenePaths(
        image=base / io_cfg.input_subdir / _scene_image_filename(scene_number, io_cfg),
        raw_video=base
        / io_cfg.raw_videos_subdir
        / _scene_video_filename(scene_number, io_cfg),
        refined_video=base
        / io_cfg.refined_videos_subdir
        / _scene_video_filename(scene_number, io_cfg),
    )


def script_base_path(io_cfg: InputOutputConfig) -> Path | None:
    base = _resolve_path(io_cfg.script_path)
    return base if base.is_dir() else None


def scene_paths_for_script(
    script_id: UUID | str,
    io_cfg: InputOutputConfig,
) -> list[ScenePaths]:
    """All scene paths for one script, in scene order."""
    results: list[ScenePaths] = []
    scene_number = 0
    while True:
        paths = scene_paths(script_id, scene_number, io_cfg)
        if not paths.image.exists():
            break
        results.append(paths)
        scene_number += 1
    return results


def scene_paths_by_script(
    io_cfg: InputOutputConfig,
) -> dict[str, list[ScenePaths]]:
    """Map each script UUID under ``io_cfg.script_path`` to its scene paths."""
    base = script_base_path(io_cfg)
    if base is None:
        return {}

    result: dict[str, list[ScenePaths]] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        try:
            UUID(entry.name)
        except ValueError:
            continue
        scenes = scene_paths_for_script(entry.name, io_cfg)
        if scenes:
            result[entry.name] = scenes
    return result