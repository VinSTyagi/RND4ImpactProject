from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from PIL import Image


import yaml
import json

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


def idea_prompt_payload(idea: dict[str, Any]) -> dict[str, str]:
    """Story fields only, for stage 2+ prompts."""
    return {name: str(idea[name]) for name in _IDEA_FIELDS}


@dataclass
class Script:
    idea: dict[str, Any]
    model: str
    script_id: UUID = field(default_factory=uuid4)
    raw_title: str | None = None
    script_scenes: list[str] | None = None
    images: list[Image] | None = None

    @classmethod
    def parse_idea_dict(cls, data: Any) -> dict[str, Any]:
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
        idea: dict[str, Any] = {
            name: str(data[name]).strip() for name in _IDEA_FIELDS
        }
        idea["model"] = str(data.get("model") or "").strip()
        return idea

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
        script_scenes = data.get("script_scenes")
        if script_scenes is not None and not isinstance(script_scenes, list):
            raise ValueError("script_scenes must be a list or null")
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


@dataclass
class VLLMModelConfig:
    model_path: str = "Qwen/Qwen3-4B-AWQ"
    quantization: str = "awq"
    max_tokens: int = 1536
    temperature: float = 0.25
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    enable_thinking: bool = True
    max_model_len: int = 2048
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.80
    enforce_eager: bool = True
    batch_size: int = 1
    max_num_batched_tokens: int = 8192


@dataclass
class IdeaConfig:
    num_ideas: int = 5
    prompt_path: str = "script_setup/prompts/stage_1.md"
    output_path: str = "data/"


@dataclass
class TitleConfig:
    num_words: int = 10
    max_retries: int = 3
    prompt_path: str = "script_setup/prompts/stage_2.md"
    script_path: str = "data/"
    output_path: str = "data/"


@dataclass
class SceneConfig:
    num_scenes: int = 8
    prompt_path: str = "script_setup/prompts/stage_3.md"
    script_path: str = "data/"


@dataclass
class ImageConfig:
    prompt_path: str = "script_setup/prompts/stage_4.md"
    scene_path: str = "data/"
    output_path: str = "data/"


@dataclass
class VideoConfig:
    scene_path: str = "data/"
    image_path: str = "data/"
    output_path: str = "data/"


@dataclass
class PipelineConfig:
    stage_1_vllm_config: VLLMModelConfig = field(default_factory=VLLMModelConfig)
    stage_2_vllm_config: VLLMModelConfig = field(default_factory=VLLMModelConfig)
    idea_config: IdeaConfig = field(default_factory=IdeaConfig)
    title_config: TitleConfig = field(default_factory=TitleConfig)
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    image_config: ImageConfig = field(default_factory=ImageConfig)
    video_config: VideoConfig = field(default_factory=VideoConfig)


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


def load_config(path: str) -> PipelineConfig:
    data = _load_yaml(path)
    return PipelineConfig(
        stage_1_vllm_config=VLLMModelConfig(**_section(data, "stage_1_vllm_config")),
        stage_2_vllm_config=VLLMModelConfig(**_section(data, "stage_2_vllm_config")),
        idea_config=IdeaConfig(**_section(data, "idea_config")),
        title_config=TitleConfig(**_section(data, "title_config")),
        scene_config=SceneConfig(**_section(data, "scene_config")),
        image_config=ImageConfig(**_section(data, "image_config")),
        video_config=VideoConfig(**_section(data, "video_config")),
    )
