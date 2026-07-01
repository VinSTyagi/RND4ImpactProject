from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml


@dataclass
class TTSModelConfig:
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    model_family: str = "Qwen"
    dtype: str = "auto"
    device: str = "cuda"
    quantization: str = "none"
    max_model_len: int = 2048
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.80
    enforce_eager: bool = True
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 8192
    input_subdir: str = "data/"
    output_subdir: str = "audio/"
    audio_template: str = "scene_{scene_number:02d}_{line_index:02d}.wav"
    save_raw: bool = True
    skip_existing: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TTSModelConfig:
        return cls(**_dataclass_kwargs(cls, data))

    @classmethod
    def from_yaml(cls, path: str) -> TTSModelConfig:
        data = _load_yaml(path)
        return cls.from_dict(data)


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


def load_config(path: str) -> TTSModelConfig:
    data = _load_yaml(path)
    tts_kwargs = _dataclass_kwargs(TTSModelConfig, _section(data, "tts_model_config"))
    io_kwargs = _dataclass_kwargs(TTSModelConfig, _section(data, "io_config"))
    return TTSModelConfig(**{**tts_kwargs, **io_kwargs})


def scene_audio_output_path(
    script_id: UUID | str,
    scene_number: int,
    line_index: int,
    config: TTSModelConfig,
) -> Path:
    """Path for one dialogue beat: ``.../audio/scene_<scene>_<line>.wav``."""
    from utils.schema import resolve_path

    filename = config.audio_template.format(
        scene_number=scene_number,
        line_index=line_index,
    )
    return (
        resolve_path(config.input_subdir)
        / str(script_id)
        / str(scene_number)
        / config.output_subdir
        / filename
    )
