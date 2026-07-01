from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VLLMModelConfig:
    model_path: str = "Qwen/Qwen3-4B-AWQ"
    quantization: str = "awq"
    dtype: str = "auto"
    max_tokens: int = 1536
    temperature: float = 0.25
    top_p: float = 0.9
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    enable_thinking: bool = True
    max_model_len: int = 2048
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.80
    enforce_eager: bool = True
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 8192


@dataclass
class IdeaConfig:
    num_ideas: int = 5
    prompt_path: str = "script_setup/prompts/stage_1.md"
    output_path: str = "data/"
    batch_size: int = 0


@dataclass
class TitleConfig:
    num_words: int = 10
    prompt_path: str = "script_setup/prompts/stage_2.md"
    script_path: str = "data/"
    batch_size: int = 0


@dataclass
class SceneOutlineConfig:
    num_scenes: int = 5
    prompt_path: str = "script_setup/prompts/stage_3.md"
    script_path: str = "data/"
    batch_size: int = 1  # scripts per vLLM call; <=0 = all scripts at once


@dataclass
class SceneContentConfig:
    min_beats: int = 10
    max_beats: int = 15
    prompt_path: str = "script_setup/prompts/stage_4.md"
    script_path: str = "data/"
    batch_size: int = 1  # scenes per vLLM batch; <=0 = all scenes at once


@dataclass
class ImagePromptConfig:
    min_prompts: int = 1  # minimum SDXL image prompts per scene
    max_prompts: int = 5  # maximum SDXL image prompts per scene
    prompt_path: str = "script_setup/prompts/stage_5.md"
    script_path: str = "data/"
    batch_size: int = 0  # scenes per vLLM batch; <=0 = all scenes at once


@dataclass
class PipelineConfig:
    global_vllm_config: VLLMModelConfig = field(default_factory=VLLMModelConfig)
    idea_config: IdeaConfig = field(default_factory=IdeaConfig)
    title_config: TitleConfig = field(default_factory=TitleConfig)
    scene_outline_config: SceneOutlineConfig = field(default_factory=SceneOutlineConfig)
    scene_content_config: SceneContentConfig = field(default_factory=SceneContentConfig)
    image_config: ImagePromptConfig = field(default_factory=ImagePromptConfig)


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


def load_config(path: str) -> PipelineConfig:
    data = _load_yaml(path)
    scene_outline_section = _section(data, "scene_outline_config") or _section(
        data, "scene_config"
    )
    scene_outline_section_kwargs = _dataclass_kwargs(
        SceneOutlineConfig, scene_outline_section
    )
    return PipelineConfig(
        global_vllm_config=VLLMModelConfig(
            **_dataclass_kwargs(VLLMModelConfig, _section(data, "global_vllm_config"))
        ),
        idea_config=IdeaConfig(
            **_dataclass_kwargs(IdeaConfig, _section(data, "idea_config"))
        ),
        title_config=TitleConfig(
            **_dataclass_kwargs(TitleConfig, _section(data, "title_config"))
        ),
        scene_outline_config=SceneOutlineConfig(**scene_outline_section_kwargs),
        scene_content_config=SceneContentConfig(
            **_dataclass_kwargs(SceneContentConfig, _section(data, "scene_content_config"))
        ),
        image_config=ImagePromptConfig(
            **_dataclass_kwargs(ImagePromptConfig, _section(data, "image_config"))
        ),
    )
