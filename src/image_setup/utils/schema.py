from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

from PIL import Image

import yaml

from utils.image_prompt import (
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


_IDEA_FIELDS = (
    "genre",
    "setting",
    "premise",
    "protagonist",
    "antagonist",
    "hook",
    "tone",
    "theme",
)


class Idea(TypedDict):
    genre: str
    setting: str
    premise: str
    protagonist: str
    antagonist: str
    hook: str
    tone: str
    theme: str
    model: str


def idea_prompt_payload(idea: Idea) -> dict[str, str]:
    """Story fields only, for stage 2+ prompts."""
    return {name: idea[name] for name in _IDEA_FIELDS}


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
    image_prompt: ImagePrompt | None


def scene_payload(scene: Scene) -> dict[str, str]:
    return {name: scene[name] for name in _SCENE_FIELDS}


@dataclass
class Script:
    idea: Idea
    model: str
    script_id: UUID = field(default_factory=uuid4)
    raw_title: str | None = None
    script_scenes: list[Scene] | None = None
    images: list[Image] | None = None

    @classmethod
    def parse_idea_dict(cls, data: Any) -> Idea:
        """Validate and normalize an idea dict from LLM output or JSON."""
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        missing = [
            name
            for name in _IDEA_FIELDS
            if name not in data or not str(data[name]).strip()
        ]
        if missing:
            raise ValueError(f"missing or empty fields: {', '.join(missing)}")
        idea: Idea = {
            "genre": str(data["genre"]).strip(),
            "setting": str(data["setting"]).strip(),
            "premise": str(data["premise"]).strip(),
            "protagonist": str(data["protagonist"]).strip(),
            "antagonist": str(data["antagonist"]).strip(),
            "hook": str(data["hook"]).strip(),
            "tone": str(data["tone"]).strip(),
            "theme": str(data["theme"]).strip(),
            "model": str(data.get("model") or "").strip(),
        }
        return idea

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

        image_prompt_data = data.get("image_prompt")
        if image_prompt_data is None:
            scene_image_prompt = None
        else:
            scene_image_prompt = cls.parse_img_prompt_dict(image_prompt_data)

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
            "image_prompt": scene_image_prompt,
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
    def parse_scenes_list(cls, data: Any) -> list[Scene] | None:
        """Validate script_scenes from JSON (dicts or legacy JSON strings)."""
        if data is None:
            return None
        if not isinstance(data, list):
            raise ValueError("script_scenes must be a list or null")

        scenes: list[Scene] = []
        for index, item in enumerate(data):
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"script_scenes[{index}] is not valid JSON: {exc}"
                    ) from exc
            try:
                scenes.append(cls.parse_scene_dict(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"script_scenes[{index}]: {exc}") from exc
        return scenes

    @classmethod
    def parse_img_prompt_dict(cls, data: Any) -> ImagePrompt:
        """Validate and normalize an image prompt dict from LLM output or JSON."""
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")

        missing = [name for name in IMAGE_PROMPT_FIELDS if name not in data]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")

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
        )
        return normalized  # type: ignore[return-value]

    @classmethod
    def attach_image_prompts(
        cls, scenes: list[Scene], image_prompts: list[ImagePrompt]
    ) -> list[Scene]:
        """Return scenes with image_prompt set from a parallel prompt list."""
        if len(scenes) != len(image_prompts):
            raise ValueError(
                f"expected {len(scenes)} image prompt(s) but received "
                f"{len(image_prompts)}"
            )
        updated: list[Scene] = []
        for scene, image_prompt in zip(scenes, image_prompts):
            updated.append({**scene, "image_prompt": image_prompt})
        return updated

    def prompt_payload(self) -> dict[str, str]:
        """Story fields plus title, for stage 3 prompts."""
        if not self.raw_title:
            raise ValueError(f"script {self.script_id} missing raw_title")
        payload = idea_prompt_payload(self.idea)
        payload["title"] = self.raw_title
        return payload

    def to_json(self) -> dict[str, Any]:
        return {
            "idea": self.idea,
            "script_id": str(self.script_id),
            "raw_title": self.raw_title,
            "script_scenes": self.script_scenes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Script:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        idea_data = data.get("idea") or data.get("raw_idea")
        if not isinstance(idea_data, dict):
            raise ValueError("missing or invalid idea")
        idea = cls.parse_idea_dict(idea_data)
        model = str(data.get("model") or idea.get("model") or "").strip()
        script_id_raw = data.get("script_id") or idea_data.get("idea_id")
        if not script_id_raw:
            raise ValueError("missing script_id")
        raw_title = data.get("raw_title")
        if raw_title is not None:
            raw_title = str(raw_title).strip() or None
        script_scenes = cls.parse_scenes_list(data.get("script_scenes"))
        return cls(
            idea=idea,
            model=model,
            script_id=UUID(str(script_id_raw)),
            raw_title=raw_title,
            script_scenes=script_scenes,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> Script:
        resolved = Path(path) if isinstance(path, Path) else resolve_path(str(path))
        if not resolved.exists():
            raise FileNotFoundError(f"script file not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def read_all(cls, data_root: str) -> list[Script]:
        """Load one Script per ``<script_id>/script.json`` under data_root."""
        base = resolve_path(data_root)
        if not base.exists():
            raise FileNotFoundError(f"data root not found: {base}")
        scripts: list[Script] = []
        for entry in sorted(base.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            script_path = entry / "script.json"
            if script_path.is_file():
                scripts.append(cls.load_json(script_path))
        if not scripts:
            raise ValueError(f"no scripts found in {base}")
        return scripts

    def save(self, data_root: str) -> None:
        out = self.to_json()
        try:
            out_dir = resolve_path(data_root) / str(self.script_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "script.json").open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception:
            raise


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


@dataclass
class OutputConfig:
    script_path: str = "data/"
    raw_subdir: str = "raw_images"
    output_subdir: str = "refined_images"
    filename_template: str = "scene_{scene_number:02d}.png"
    save_raw: bool = True
    skip_existing: bool = True


@dataclass
class RefinementConfig:
    enabled: bool = True
    type: str = "sdxl_refiner"
    model_path: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    variant: str | None = "fp16"
    scheduler: str = "euler"
    num_inference_steps: int = 20
    denoising_start: float = 0.8
    denoising_end: float = 0.8
    strength: float = 0.35


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


def load_config(path: str) -> ImageSetupPipelineConfig:
    data = _load_yaml(path)
    return ImageSetupPipelineConfig(
        pipeline_config=DiffusionPipelineConfig(**_section(data, "pipeline_config")),
        quantization_config=QuantizationConfig(**_section(data, "quantization_config")),
        generation_config=GenerationConfig(**_section(data, "generation_config")),
        refinement_config=RefinementConfig(**_section(data, "refinement_config")),
        output_config=OutputConfig(**_section(data, "output_config")),
    )


def parse_cfg_scale(value: str, default: float) -> float:
    raw = str(value).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid cfg_scale: {value!r}") from exc


def validate_pipeline_config(config: ImageSetupPipelineConfig) -> None:
    """Reject incompatible pipeline and refinement combinations."""
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
        return

    ref_type = config.refinement_config.type.strip().lower()
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


def format_positive_prompt(tags: list[str]) -> str:
    return join_prompt_tags(truncate_tags_to_clip(tags))


def format_negative_prompt(tags: list[str]) -> str:
    return join_prompt_tags(truncate_tags_to_clip(tags))


def _scene_filename(scene_number: int, output_cfg: OutputConfig) -> str:
    return output_cfg.filename_template.format(scene_number=scene_number)


def scene_output_path(
    script_id: UUID | str,
    scene_number: int,
    output_cfg: OutputConfig,
) -> Path:
    return (
        resolve_path(output_cfg.script_path)
        / str(script_id)
        / output_cfg.output_subdir
        / _scene_filename(scene_number, output_cfg)
    )


def scene_raw_output_path(
    script_id: UUID | str,
    scene_number: int,
    output_cfg: OutputConfig,
) -> Path:
    return (
        resolve_path(output_cfg.script_path)
        / str(script_id)
        / output_cfg.raw_subdir
        / _scene_filename(scene_number, output_cfg)
    )

