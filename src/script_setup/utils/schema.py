from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from PIL import Image

import re

import yaml
import json
import os

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


@dataclass
class Idea:
    genre: str
    setting: str
    premise: str
    protagonist: str
    antagonist: str
    hook: str
    tone: str
    theme: str
    idea_id: UUID = field(default_factory=uuid4)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Idea:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        missing = [
            name
            for name in _IDEA_FIELDS
            if name not in data or not str(data[name]).strip()
        ]
        if missing:
            raise ValueError(f"missing or empty fields: {', '.join(missing)}")
        fields = {name: str(data[name]).strip() for name in _IDEA_FIELDS}
        idea_id = data.get("idea_id")
        if idea_id:
            fields["idea_id"] = UUID(str(idea_id))
        return cls(**fields)

    def to_json(self) -> dict[str, Any]:
        return {
            "genre": self.genre,
            "setting": self.setting,
            "premise": self.premise,
            "protagonist": self.protagonist,
            "antagonist": self.antagonist,
            "hook": self.hook,
            "tone": self.tone,
            "theme": self.theme,
            "idea_id": str(self.idea_id),
        }


@dataclass
class Script:
    raw_idea: Idea
    script_id: UUID = field(init=False)
    raw_title: str | None = None
    script_scenes: list[str] | None = None
    images: list[Image] | None = None

    def __post_init__(self) -> None:
        self.script_id = self.raw_idea.idea_id

    def to_json(self):
        return {
            "raw_idea": self.raw_idea.to_json(),
            "script_id": self.script_id,
            "raw_title": self.raw_title,
            "script_scenes": self.script_scenes,
            "num_images": len(self.images),
        }

    def save(self, script_folder: str) -> None:
        out = self.to_json()
        try:
            out_dir = os.path.join(script_folder, self.script_id)
            os.makedirs(out_dir, exist_ok=True)
            with open(out_dir, "w") as f:
                f.write(json.dump(out))
        except Exception:
            raise


@dataclass
class VLLMModelConfig:
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens: int = 12800
    temperature: float = 0.25
    max_model_len: int = 32768
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    batch_size: int = 4


@dataclass
class IdeaConfig:
    num_ideas: int = 5
    prompt_path: str = "script_setup/prompts/stage_1.md"
    output_path: str = "script_setup/output/stage_1.jsonl"


@dataclass
class TitleConfig:
    prompt_path: str = "script_setup/prompts/stage_2.md"
    idea_path: str = "script_setup/output/stage_1.jsonl"
    output_path: str = "script_setup/output/script/stage_2.jsonl"


@dataclass
class SceneConfig:
    num_scenes: int = 8
    prompt_path: str = "script_setup/prompts/stage_3.md"
    script_path: str = "script_setup/output/script/"


@dataclass
class ImageConfig:
    prompt_path: str = "script_setup/prompts/stage_4.md"
    scene_path: str = "script_setup/output/script/"
    output_path: str = "script_setup/output/images/"


@dataclass
class PipelineConfig:
    vllm_model_config: VLLMModelConfig = field(default_factory=VLLMModelConfig)
    idea_config: IdeaConfig = field(default_factory=IdeaConfig)
    title_config: TitleConfig = field(default_factory=TitleConfig)
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    image_config: ImageConfig = field(default_factory=ImageConfig)


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
        vllm_model_config=VLLMModelConfig(**_section(data, "vllm_model_config")),
        idea_config=IdeaConfig(**_section(data, "idea_config")),
        title_config=TitleConfig(**_section(data, "title_config")),
        scene_config=SceneConfig(**_section(data, "scene_config")),
        image_config=ImageConfig(**_section(data, "image_config")),
    )
