from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VLLMModelConfig:
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_new_tokens: int = 6400
    temperature: float = 0.2
    max_model_len: int = 32768
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    batch_size: int = 8


@dataclass
class IdeaConfig:
    num_ideas: int = 10
    prompt_path: str = "prompts/stage_1.md"
    output_path: str = "output/stage_1.jsonl"


@dataclass
class TitleConfig:
    prompt_path: str = "prompts/stage_2.md"
    idea_path: str = "output/stage_1.jsonl"
    output_path: str = "output/stage_2.jsonl"


@dataclass
class SceneConfig:
    prompt_path: str = "prompts/stage_3.md"
    title_path: str = "output/stage_2.jsonl"
    output_path: str = "output/stage_3.jsonl"


@dataclass
class ImageConfig:
    prompt_path: str = "prompts/stage_4.md"
    scene_path: str = "output/stage_3.jsonl"
    output_path: str = "output/stage_4.jsonl"


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


def load_config(path: str) -> PipelineConfig:
    data = _load_yaml(path)
    return PipelineConfig(
        vllm_model_config=VLLMModelConfig(**data.get("vllm_model_config", {})),
        idea_config=IdeaConfig(**data.get("idea_config", {})),
        title_config=TitleConfig(**data.get("title_c onfig", {})),
        scene_config=SceneConfig(**data.get("scene_config", {})),
        image_config=ImageConfig(**data.get("image_config", {})),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage_1.yaml")
    args = parser.parse_args()
    print(load_config(args.config))
