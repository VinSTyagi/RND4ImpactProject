from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

from PIL import Image

import yaml

from utils.image_prompt import (
    coerce_line_indices,
    coerce_prompt_tags,
    join_prompt_tags,
    normalize_image_prompt_fields,
    truncate_tags_to_clip,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(rel: str) -> Path:
    """Resolve config paths: ``data/...`` from repo root, else from ``src/``."""
    path = Path(rel)
    if path.is_absolute():
        return path
    normalized = rel.replace("\\", "/")
    if normalized.startswith("data/"):
        return _REPO_ROOT / path
    return _SRC_ROOT / path



IMAGE_PROMPT_FIELDS = (
    "positive_prompt",
    "negative_prompt",
    "style_preset",
    "aspect_ratio",
    "cfg_scale",
    "reasoning",
)

_SCENE_FIELDS = (
    "scene_number",
    "scene_title",
    "act",
    "setting",
    "characters",
    "summary",
    "conflict",
    "emotional_beat",
    "character_change",
    "ends_on",
)


_VALID_ACTS = frozenset(
    {"setup", "rising_action", "climax", "falling_action", "resolution"}
)


class ImagePrompt(TypedDict):
    positive_prompt: list[str]
    negative_prompt: list[str]
    style_preset: str
    aspect_ratio: str
    cfg_scale: str
    reasoning: str
    lines_used: list[int]


class Scene(TypedDict):
    scene_number: int
    scene_title: str
    act: str
    setting: str
    characters: list[str]
    summary: str
    conflict: str
    emotional_beat: str
    character_change: str
    ends_on: str


def parse_scene_content(data: Any) -> list[tuple[str, str]]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError("scene_content must be an array")
    content: list[tuple[str, str]] = []
    for index, item in enumerate(data):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            character = str(item[0]).strip()
            text = str(item[1])
            if not character:
                raise ValueError(f"scene_content[{index}]: character name must be non-empty")
            content.append((character, text))
            continue
        raise ValueError(f"scene_content[{index}] must be a 2-element array")
    return content


def scene_payload(scene: Scene) -> dict[str, str]:
    return {name: scene[name] for name in _SCENE_FIELDS}


@dataclass
class SceneScript:
    """One scene loaded from ``<script_id>/<n>/script.json``."""

    script_id: UUID
    model: str
    scene: Scene
    scene_content: list[tuple[str, str]] = field(default_factory=list)
    image_prompt: list[ImagePrompt] | None = None

    @classmethod
    def parse_scene_dict(cls, data: Any) -> Scene:
        """Validate and normalize a scene dict from LLM output or JSON."""
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")

        missing = [name for name in _SCENE_FIELDS if name not in data]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")

        act = str(data["act"]).strip()
        if act not in _VALID_ACTS:
            raise ValueError(f"invalid act: {act}")

        characters = data["characters"]
        if not isinstance(characters, list) or not characters:
            raise ValueError("characters must be a non-empty array")

        scene_number = data["scene_number"]
        if not isinstance(scene_number, int):
            raise TypeError("scene_number must be an integer")

        scene_characters = [
            str(name).strip() for name in characters if str(name).strip()
        ]
        if not scene_characters:
            raise ValueError("characters must contain at least one name")

        scene: Scene = {
            "scene_number": scene_number,
            "act": act,
            "characters": scene_characters,
            "scene_title": str(data["scene_title"]).strip(),
            "setting": str(data["setting"]).strip(),
            "summary": str(data["summary"]).strip(),
            "conflict": str(data["conflict"]).strip(),
            "emotional_beat": str(data["emotional_beat"]).strip(),
            "character_change": str(data["character_change"]).strip(),
            "ends_on": str(data["ends_on"]).strip(),
        }
        for name in (
            "scene_title",
            "setting",
            "summary",
            "conflict",
            "emotional_beat",
            "character_change",
            "ends_on",
        ):
            if not scene[name]:
                raise ValueError(f"{name} must be a non-empty string")
        return scene

    @classmethod
    def parse_img_prompt_dict(
        cls,
        data: Any,
        *,
        beat_count: int = 0,
        require_lines_used: bool = False,
    ) -> ImagePrompt:
        """Validate and normalize an image prompt dict from LLM output or JSON."""
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")

        missing = [name for name in IMAGE_PROMPT_FIELDS if name not in data]
        if require_lines_used and "lines_used" not in data:
            missing.append("lines_used")
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")

        if "lines_used" in data or require_lines_used:
            lines_used = coerce_line_indices(
                data.get("lines_used"),
                beat_count=beat_count,
                allow_empty=not require_lines_used,
            )
        else:
            lines_used = []

        normalized = normalize_image_prompt_fields(
            positive_prompt=coerce_prompt_tags(
                data["positive_prompt"], "positive_prompt"
            ),
            negative_prompt=coerce_prompt_tags(
                data["negative_prompt"], "negative_prompt"
            ),
            style_preset=data["style_preset"],
            aspect_ratio=data["aspect_ratio"],
            cfg_scale=data["cfg_scale"],
            reasoning=data["reasoning"],
            lines_used=lines_used,
        )
        return normalized  # type: ignore[return-value]

    @classmethod
    def parse_img_prompt_list(
        cls,
        data: Any,
        *,
        beat_count: int = 0,
    ) -> list[ImagePrompt] | None:
        if data is None:
            return None
        parse_kwargs = {"beat_count": beat_count}
        if isinstance(data, dict):
            return [cls.parse_img_prompt_dict(data, **parse_kwargs)]
        if isinstance(data, list):
            if not data:
                raise ValueError("image_prompt must be null or a non-empty array")
            return [cls.parse_img_prompt_dict(item, **parse_kwargs) for item in data]
        raise TypeError(
            f"image_prompt must be an object, array, or null, got {type(data).__name__}"
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "script_id": str(self.script_id),
            "model": self.model,
            "scene": dict(self.scene),
            "scene_content": [[character, text] for character, text in self.scene_content],
        }
        if self.image_prompt is not None:
            out["image_prompt"] = self.image_prompt
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneScript:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        script_id_raw = data.get("script_id")
        if not script_id_raw:
            raise ValueError("missing script_id")
        model = str(data.get("model") or "").strip()
        scene_data = data.get("scene")
        if scene_data is None:
            raise ValueError("missing scene")
        scene = cls.parse_scene_dict(scene_data)
        scene_content = parse_scene_content(data.get("scene_content"))
        image_prompt_raw = data.get("image_prompt")
        if image_prompt_raw is None and isinstance(scene_data, dict):
            image_prompt_raw = scene_data.get("image_prompt")
        image_prompt = cls.parse_img_prompt_list(
            image_prompt_raw,
            beat_count=len(scene_content),
        )
        return cls(
            script_id=UUID(str(script_id_raw)),
            model=model,
            scene=scene,
            scene_content=scene_content,
            image_prompt=image_prompt,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> SceneScript:
        resolved = Path(path) if isinstance(path, Path) else resolve_path(str(path))
        if not resolved.is_file():
            raise FileNotFoundError(f"scene script file not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def read_all(cls, data_root: str) -> list[SceneScript]:
        """Load every ``<script_id>/<scene>/script.json`` under data_root."""
        base = resolve_path(data_root)
        if not base.is_dir():
            raise FileNotFoundError(f"data root not found: {base}")
        scripts: list[SceneScript] = []
        for story_path in sorted(base.iterdir(), key=lambda p: p.name):
            if not story_path.is_dir():
                continue
            for scene_path in sorted(
                (
                    entry
                    for entry in story_path.iterdir()
                    if entry.is_dir() and entry.name.isdigit()
                ),
                key=lambda path: int(path.name),
            ):
                script_json = scene_path / "script.json"
                if script_json.is_file():
                    scripts.append(cls.load_json(script_json))
        if not scripts:
            raise ValueError(f"no scene scripts found in {base}")
        scripts.sort(
            key=lambda item: (str(item.script_id), item.scene["scene_number"])
        )
        return scripts


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
    # Optional distilled UNet weights (e.g. ByteDance/SDXL-Lightning checkpoints).
    unet_checkpoint_repo: str | None = None
    unet_checkpoint_file: str | None = None

    def uses_distilled_low_cfg(self) -> bool:
        """True for turbo/lightning-style models that need low guidance scale."""
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
    # Total scheduler steps shared by base (denoising_end) and refiner (denoising_start).
    num_inference_steps: int = 40
    denoising_start: float = 0.8
    denoising_end: float = 0.8
    # Used only when refining a fully decoded PNG from disk (stage 2 alone).
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
    """True when stage 2 should run a refinement backend."""
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
    """Reject incompatible pipeline combinations."""
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
    """Return output size from generation_config or fall back to scene aspect_ratio."""
    if gen_cfg.width is None and gen_cfg.height is None:
        return resolve_aspect_size(image_prompt["aspect_ratio"], pipeline_type)
    if gen_cfg.width is None or gen_cfg.height is None:
        raise ValueError(
            "generation_config width and height must both be set when overriding "
            "scene aspect_ratio"
        )
    return int(gen_cfg.width), int(gen_cfg.height)


def format_positive_prompt(tags: list[str]) -> str:
    return join_prompt_tags(truncate_tags_to_clip(tags))


def format_negative_prompt(tags: list[str]) -> str:
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
