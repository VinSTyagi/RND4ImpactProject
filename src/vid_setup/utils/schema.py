from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from uuid import UUID
import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = Path(__file__).resolve().parents[2]

_SUPPORTED_PIPELINE_TYPES = frozenset({"svd", "ltx", "sana", "cogvideox", "wan"})
_SUPPORTED_UPSCALE_TYPES = frozenset({"none", "ltx_latent"})
_PROMPTED_PIPELINE_TYPES = frozenset({"ltx", "sana", "cogvideox", "wan"})

# Diffusers pipeline classes used by the wrapper for each backend type.
_PIPELINE_CLASS_BY_TYPE: dict[str, str] = {
    "svd": "StableVideoDiffusionPipeline",
    "ltx": "LTXImageToVideoPipeline",
    "sana": "SanaImageToVideoPipeline",
    "cogvideox": "CogVideoXImageToVideoPipeline",
    "wan": "WanImageToVideoPipeline",
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


_SVD_GENERATION_FIELDS = frozenset(
    {
        "min_guidance_scale",
        "max_guidance_scale",
        "fps",
        "motion_bucket_id",
        "noise_aug_strength",
        "decode_chunk_size",
    }
)
_LTX_GENERATION_FIELDS = frozenset(
    {
        "guidance_scale",
        "frame_rate",
        "decode_timestep",
        "decode_noise_scale",
        "prompt",
        "negative_prompt",
    }
)
_SANA_GENERATION_FIELDS = frozenset(
    {
        "guidance_scale",
        "prompt",
        "negative_prompt",
        "use_resolution_binning",
    }
)
_COGVIDEOX_GENERATION_FIELDS = frozenset(
    {
        "guidance_scale",
        "prompt",
        "negative_prompt",
        "use_dynamic_cfg",
    }
)
_WAN_GENERATION_FIELDS = frozenset(
    {
        "guidance_scale",
        "prompt",
        "negative_prompt",
    }
)

# YAML fields used only when writing the output mp4 (not pipeline __call__ kwargs).
_EXPORT_ONLY_FIELDS_BY_TYPE: dict[str, frozenset[str]] = {
    "svd": frozenset(),
    "ltx": frozenset(),
    "sana": frozenset({"frame_rate"}),
    "cogvideox": frozenset({"frame_rate"}),
    "wan": frozenset({"frame_rate"}),
}

_SHARED_GENERATION_FIELDS = frozenset(
    {
        "num_inference_steps",
        "num_frames",
    }
)

_GENERATION_FIELDS_BY_TYPE: dict[str, frozenset[str]] = {
    "svd": _SHARED_GENERATION_FIELDS | _SVD_GENERATION_FIELDS,
    "ltx": _SHARED_GENERATION_FIELDS | _LTX_GENERATION_FIELDS,
    "sana": _SHARED_GENERATION_FIELDS | _SANA_GENERATION_FIELDS,
    "cogvideox": _SHARED_GENERATION_FIELDS | _COGVIDEOX_GENERATION_FIELDS,
    "wan": _SHARED_GENERATION_FIELDS | _WAN_GENERATION_FIELDS,
}


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
    prompt: str | None = None
    negative_prompt: str | None = None

    # Sana-specific (optional)
    use_resolution_binning: bool | None = None

    # CogVideoX-specific (optional)
    use_dynamic_cfg: bool | None = None


def pipeline_needs_prompt(pipeline_type: str) -> bool:
    return pipeline_type.strip().lower() in _PROMPTED_PIPELINE_TYPES


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
    enable_group_offload: bool = False


@dataclass
class InputOutputConfig:
    """Script-relative paths for scene images and generated videos."""

    script_path: str = "data/"
    input_subdir: str = "refined_images"
    image_template: str = "scene_{scene_number}_{prompt_number}.png"
    raw_videos_subdir: str = "raw_videos"
    refined_videos_subdir: str = "refined_videos"
    video_template: str = "scene_{scene_number}_{prompt_number}.mp4"
    save_raw: bool = True
    skip_existing: bool = True


def _coerce_prompt_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    return []


def format_prompt_tags(tags: list[str]) -> str:
    return ", ".join(tag for tag in tags if tag)


@dataclass
class Script:
    """Script metadata loaded from ``data/<script_id>/script.json``."""

    script_id: UUID
    script_scenes: list[dict[str, Any]] | None = None
    raw_title: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Script:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")

        script_id_raw = data.get("script_id")
        if not script_id_raw:
            idea = data.get("idea") or data.get("raw_idea") or {}
            if isinstance(idea, dict):
                script_id_raw = idea.get("idea_id")
        if not script_id_raw:
            raise ValueError("script.json missing script_id")

        raw_scenes = data.get("script_scenes")
        script_scenes: list[dict[str, Any]] | None = None
        if raw_scenes is not None:
            if not isinstance(raw_scenes, list):
                raise ValueError("script_scenes must be a list or null")
            script_scenes = []
            for index, item in enumerate(raw_scenes):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"script_scenes[{index}] is not valid JSON: {exc}"
                        ) from exc
                if not isinstance(item, dict):
                    raise ValueError(
                        f"script_scenes[{index}] must be an object, "
                        f"got {type(item).__name__}"
                    )
                script_scenes.append(item)

        raw_title = data.get("raw_title")
        if raw_title is not None:
            raw_title = str(raw_title).strip() or None

        return cls(
            script_id=UUID(str(script_id_raw)),
            script_scenes=script_scenes,
            raw_title=raw_title,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> Script:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"script file not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def load_for_script_id(
        cls,
        script_id: UUID | str,
        io_cfg: InputOutputConfig,
    ) -> Script:
        script_path = _resolve_path(io_cfg.script_path) / str(script_id) / "script.json"
        return cls.load_json(script_path)

    @classmethod
    def read_all(cls, data_root: str) -> list[Script]:
        """Load one Script per ``<script_id>/script.json`` under data_root."""
        base = _resolve_path(data_root)
        if not base.is_dir():
            raise FileNotFoundError(f"data root not found: {base}")
        scripts: list[Script] = []
        for entry in sorted(base.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            script_path = entry / "script.json"
            if script_path.is_file():
                scripts.append(cls.load_json(script_path))
        return scripts

    @classmethod
    def load_by_ids(
        cls,
        script_ids: list[str],
        io_cfg: InputOutputConfig,
    ) -> dict[str, Script]:
        scripts: dict[str, Script] = {}
        for script_id in script_ids:
            scripts[script_id] = cls.load_for_script_id(script_id, io_cfg)
        return scripts

    def scene_by_number(self, scene_number: int) -> dict[str, Any] | None:
        scenes = self.script_scenes or []
        for scene in scenes:
            if scene.get("scene_number") == scene_number:
                return scene
        if 0 <= scene_number < len(scenes):
            return scenes[scene_number]
        return None

    def scene_prompts(
        self,
        scene_number: int,
        prompt_number: int,
        gen_cfg: GenerationConfig,
    ) -> tuple[str, str]:
        """Positive/negative prompts for one scene prompt from image_prompt, with config fallback."""
        default_positive = (gen_cfg.prompt or "").strip()
        default_negative = (gen_cfg.negative_prompt or "").strip()

        scene = self.scene_by_number(scene_number)
        if scene is None:
            return default_positive, default_negative

        image_prompt_data = scene.get("image_prompt")
        if image_prompt_data is None:
            return default_positive, default_negative

        if isinstance(image_prompt_data, dict):
            prompts = [image_prompt_data]
        elif isinstance(image_prompt_data, list):
            prompts = image_prompt_data
        else:
            return default_positive, default_negative

        if prompt_number < 0 or prompt_number >= len(prompts):
            return default_positive, default_negative

        image_prompt = prompts[prompt_number]
        if not isinstance(image_prompt, dict):
            return default_positive, default_negative

        positive = format_prompt_tags(
            _coerce_prompt_tags(image_prompt.get("positive_prompt"))
        )
        negative = format_prompt_tags(
            _coerce_prompt_tags(image_prompt.get("negative_prompt"))
        )
        return positive or default_positive, negative or default_negative


def validate_scripts_for_video(
    log: logging.Logger,
    scripts_by_id: dict[str, Script],
    prompt_counts: dict[str, int],
    gen_cfg: GenerationConfig,
) -> None:
    """Ensure every scene prompt has a positive prompt before loading a prompted video model."""
    errors: list[str] = []
    for script_id, total_prompts in prompt_counts.items():
        script = scripts_by_id.get(script_id)
        if script is None:
            errors.append(f"{script_id}: script.json not loaded")
            continue
        checked = 0
        for scene in sorted(
            script.script_scenes or [],
            key=lambda item: item.get("scene_number", 0),
        ):
            scene_number = int(scene.get("scene_number", checked))
            image_prompt_data = scene.get("image_prompt")
            if isinstance(image_prompt_data, dict):
                prompt_items = [image_prompt_data]
            elif isinstance(image_prompt_data, list):
                prompt_items = image_prompt_data
            else:
                prompt_items = []
            for prompt_number in range(len(prompt_items)):
                positive, _ = script.scene_prompts(scene_number, prompt_number, gen_cfg)
                if not positive:
                    errors.append(
                        f"{script_id}: scene {scene_number} prompt {prompt_number} "
                        "has no positive prompt "
                        "(set image_prompt.positive_prompt in script.json or "
                        "generation_config.prompt)"
                    )
                checked += 1
        if checked != total_prompts:
            errors.append(
                f"{script_id}: expected {total_prompts} prompt(s) from paths "
                f"but script.json defines {checked}"
            )

    if errors:
        raise ValueError(
            "video generation with prompted backends requires a positive prompt "
            "for every scene prompt:\n" + "\n".join(f"  - {item}" for item in errors)
        )

    log.info(
        "Validated prompts for %s script(s), %s scene prompt(s)",
        len(scripts_by_id),
        sum(prompt_counts.values()),
    )


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
    normalized = pipeline_type.strip().lower()
    allowed = _GENERATION_FIELDS_BY_TYPE.get(normalized)
    if allowed is None:
        raise ValueError(f"unsupported pipeline type {pipeline_type!r}")

    if normalized == "svd" and (
        gen_cfg.max_guidance_scale is not None and gen_cfg.max_guidance_scale > 3.0
    ):
        raise ValueError(
            "svd pipelines expect max_guidance_scale <= 3.0 "
            f"(got {gen_cfg.max_guidance_scale})"
        )

    if normalized == "cogvideox" and gen_cfg.num_frames != 49:
        raise ValueError(
            "cogvideox pipelines require num_frames=49 "
            f"(got {gen_cfg.num_frames}); other frame counts produce mosaic/grid artifacts"
        )

    if normalized == "wan" and (gen_cfg.num_frames - 1) % 4 != 0:
        raise ValueError(
            "wan pipelines require num_frames of the form 4*k+1 "
            f"(got {gen_cfg.num_frames})"
        )

    if normalized == "svd":
        if gen_cfg.fps is None:
            raise ValueError("svd generation_config requires fps for video export")
    elif normalized in _PROMPTED_PIPELINE_TYPES:
        if gen_cfg.frame_rate is None:
            raise ValueError(
                f"{normalized} generation_config requires frame_rate for video export"
            )

    for name, value in gen_cfg.__dict__.items():
        if name in {"width", "height"} or value is None:
            continue
        allowed_with_export = allowed | _EXPORT_ONLY_FIELDS_BY_TYPE.get(
            normalized, frozenset()
        )
        if name not in allowed_with_export:
            raise ValueError(
                f"{normalized} generation_config must not set {name!r} "
                f"(allowed: {', '.join(sorted(allowed_with_export))})"
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
        pipeline_type in _PROMPTED_PIPELINE_TYPES
        and config.quantization_config.torch_dtype.lower()
        not in {
            "bfloat16",
            "bf16",
            "float16",
            "fp16",
        }
    ):
        raise ValueError(
            f"{pipeline_type} pipelines require torch_dtype float16 or bfloat16"
        )

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


def pipeline_generation_kwargs(
    pipeline_type: str, gen_cfg: GenerationConfig
) -> dict[str, Any]:
    """Build diffusers call kwargs for the active backend (excludes width/height)."""
    normalized = pipeline_type.strip().lower()
    allowed = _GENERATION_FIELDS_BY_TYPE.get(normalized)
    if allowed is None:
        raise ValueError(f"unsupported pipeline type {pipeline_type!r}")

    kwargs = {
        key: value
        for key, value in gen_cfg.__dict__.items()
        if key in allowed and value is not None
    }
    if normalized == "sana" and "num_frames" in kwargs:
        kwargs["frames"] = kwargs.pop("num_frames")
    return kwargs


def _scene_image_filename(
    scene_number: int,
    prompt_number: int,
    io_cfg: InputOutputConfig,
) -> str:
    return io_cfg.image_template.format(
        scene_number=scene_number,
        prompt_number=prompt_number,
    )


def _scene_video_filename(
    scene_number: int,
    prompt_number: int,
    io_cfg: InputOutputConfig,
) -> str:
    return io_cfg.video_template.format(
        scene_number=scene_number,
        prompt_number=prompt_number,
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


@dataclass(frozen=True)
class ScenePaths:
    scene_number: int
    prompt_number: int
    image: Path
    raw_video: Path
    refined_video: Path


def scene_paths(
    script_id: UUID | str,
    scene_number: int,
    prompt_number: int,
    io_cfg: InputOutputConfig,
) -> ScenePaths:
    """Resolved image, raw-video, and refined-video paths for one scene prompt."""
    base = _resolve_path(io_cfg.script_path) / str(script_id) / str(scene_number)
    return ScenePaths(
        scene_number=scene_number,
        prompt_number=prompt_number,
        image=base
        / io_cfg.input_subdir
        / _scene_image_filename(scene_number, prompt_number, io_cfg),
        raw_video=base
        / io_cfg.raw_videos_subdir
        / _scene_video_filename(scene_number, prompt_number, io_cfg),
        refined_video=base
        / io_cfg.refined_videos_subdir
        / _scene_video_filename(scene_number, prompt_number, io_cfg),
    )


def _prompt_count_for_scene(scene: dict[str, Any]) -> int:
    image_prompt_data = scene.get("image_prompt")
    if isinstance(image_prompt_data, dict):
        return 1
    if isinstance(image_prompt_data, list):
        return len(image_prompt_data)
    return 0


def script_base_path(io_cfg: InputOutputConfig) -> Path | None:
    base = _resolve_path(io_cfg.script_path)
    return base if base.is_dir() else None


def _scene_numbers_from_script(script_id: UUID | str, io_cfg: InputOutputConfig) -> list[int]:
    script_json = _resolve_path(io_cfg.script_path) / str(script_id) / "script.json"
    if not script_json.is_file():
        return []
    import json

    with script_json.open(encoding="utf-8") as handle:
        data = json.load(handle)
    raw_scenes = data.get("script_scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        return []
    numbers: list[int] = []
    for index, scene in enumerate(raw_scenes):
        if not isinstance(scene, dict):
            continue
        scene_number = scene.get("scene_number", index)
        try:
            numbers.append(int(scene_number))
        except (TypeError, ValueError):
            numbers.append(index)
    return sorted(dict.fromkeys(numbers))


def scene_paths_for_script(
    script_id: UUID | str,
    io_cfg: InputOutputConfig,
    script: Script | None = None,
) -> list[ScenePaths]:
    """All scene prompt paths for one script, driven by script.json."""
    if script is None:
        script = Script.load_for_script_id(script_id, io_cfg)

    results: list[ScenePaths] = []
    for scene in sorted(
        script.script_scenes or [],
        key=lambda item: item.get("scene_number", 0),
    ):
        scene_number = int(scene.get("scene_number", len(results)))
        for prompt_number in range(_prompt_count_for_scene(scene)):
            paths = scene_paths(script_id, scene_number, prompt_number, io_cfg)
            if paths.image.is_file():
                results.append(paths)
    return results


def scene_paths_by_script(
    io_cfg: InputOutputConfig,
) -> dict[str, list[ScenePaths]]:
    """Map each script UUID under ``io_cfg.script_path`` to its scene prompt paths."""
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
        script_path = entry / "script.json"
        if not script_path.is_file():
            continue
        script = Script.load_json(script_path)
        scenes = scene_paths_for_script(entry.name, io_cfg, script)
        if scenes:
            result[entry.name] = scenes
    return result
