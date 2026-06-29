from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

import yaml

from utils.image_prompt import (
    coerce_line_indices,
    coerce_prompt_tags,
    normalize_image_prompt_fields,
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


def idea_prompt_payload(story: StoryIdea) -> dict[str, str]:
    """Story fields only, for stage 2+ prompts."""
    return {name: getattr(story, name) for name in _IDEA_FIELDS}


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


def coerce_int_field(value: Any, field_name: str) -> int:
    """Coerce LLM JSON scalars into integers for schema fields."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    raise TypeError(f"{field_name} must be an integer")


def normalize_act(value: Any) -> str:
    """Normalize act labels from LLM output onto the schema enum."""
    act = re.sub(r"[\s\-]+", "_", str(value).strip().lower())
    if act not in _VALID_ACTS:
        valid = ", ".join(sorted(_VALID_ACTS))
        raise ValueError(f"invalid act: {value!r}; expected one of: {valid}")
    return act


def coerce_character_list(value: Any) -> list[str]:
    """Normalize scene character lists from LLM output."""
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError("characters must be a non-empty array")

    characters = [str(name).strip() for name in items if str(name).strip()]
    if not characters:
        raise ValueError("characters must contain at least one name")
    return characters


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


def _serialize_scene_content(
    content: list[tuple[str, str]],
) -> list[list[str]]:
    return [[character, text] for character, text in content]


def scene_outline_payload(scene: Scene) -> dict[str, Any]:
    """Outline fields only, for stage 4 prompts."""
    return {name: scene[name] for name in _SCENE_FIELDS}


def scene_image_prompt_payload(
    scene: Scene,
    scene_content: list[tuple[str, str]],
) -> dict[str, Any]:
    """Outline plus scene_content, for stage 5 image-prompt prompts."""
    payload = scene_outline_payload(scene)
    payload["scene_content"] = _serialize_scene_content(scene_content)
    return payload


def cast_visual_context(idea: StoryIdea) -> dict[str, str]:
    """Story-bible visual anchors for stage 5 (describe cast without names)."""
    return {
        "protagonist_description": idea.protagonist,
        "antagonist_description": idea.antagonist,
    }


def story_dir(data_root: str, script_id: UUID | str) -> Path:
    return resolve_path(data_root) / str(script_id)


def idea_json_path(data_root: str, script_id: UUID | str) -> Path:
    return story_dir(data_root, script_id) / "idea.json"


def scene_dir(data_root: str, script_id: UUID | str, scene_number: int) -> Path:
    return story_dir(data_root, script_id) / str(scene_number)


def scene_script_json_path(
    data_root: str, script_id: UUID | str, scene_number: int
) -> Path:
    return scene_dir(data_root, script_id, scene_number) / "script.json"


def iter_story_dirs(data_root: str) -> list[Path]:
    base = resolve_path(data_root)
    if not base.is_dir():
        return []
    return sorted(
        (entry for entry in base.iterdir() if entry.is_dir()),
        key=lambda path: path.name,
    )


def iter_scene_dirs(story_path: Path) -> list[Path]:
    return sorted(
        (
            entry
            for entry in story_path.iterdir()
            if entry.is_dir() and entry.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )


@dataclass
class StoryIdea:
    """Story bible written by stages 1–2 to ``<script_id>/idea.json``."""

    genre: str
    setting: str
    premise: str
    protagonist: str
    antagonist: str
    hook: str
    tone: str
    theme: str
    script_id: UUID = field(default_factory=uuid4)
    model: str = ""
    title: str | None = None

    @property
    def raw_title(self) -> str | None:
        return self.title

    @classmethod
    def _idea_fields_from_dict(cls, data: Any) -> dict[str, str]:
        """Validate and normalize story-bible fields from LLM output or JSON."""
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        missing = [
            name
            for name in _IDEA_FIELDS
            if name not in data or not str(data[name]).strip()
        ]
        if missing:
            raise ValueError(f"missing or empty fields: {', '.join(missing)}")
        return {name: str(data[name]).strip() for name in _IDEA_FIELDS}

    @classmethod
    def from_idea_dict(cls, data: Any, *, model: str = "") -> StoryIdea:
        """Build a StoryIdea from LLM idea output (stage 1)."""
        return cls(**cls._idea_fields_from_dict(data), model=model)

    def prompt_payload(self) -> dict[str, str]:
        """Story fields plus title, for stage 3 prompts."""
        if not self.title:
            raise ValueError(f"story {self.script_id} missing title")
        payload = idea_prompt_payload(self)
        payload["title"] = self.title
        return payload

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: getattr(self, name) for name in _IDEA_FIELDS}
        out["script_id"] = str(self.script_id)
        out["model"] = self.model
        if self.title:
            out["title"] = self.title
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoryIdea:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        nested = data.get("idea") or data.get("raw_idea")
        if isinstance(nested, dict):
            idea_fields = cls._idea_fields_from_dict(nested)
            script_id_raw = data.get("script_id") or nested.get("idea_id")
        else:
            idea_fields = cls._idea_fields_from_dict(data)
            script_id_raw = data.get("script_id") or data.get("idea_id")
        if not script_id_raw:
            raise ValueError("missing script_id")
        title = data.get("title") or data.get("raw_title")
        if title is not None:
            title = str(title).strip() or None
        model = str(data.get("model") or "").strip()
        return cls(
            **idea_fields,
            script_id=UUID(str(script_id_raw)),
            model=model,
            title=title,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> StoryIdea:
        resolved = Path(path) if isinstance(path, Path) else resolve_path(str(path))
        if not resolved.is_file():
            raise FileNotFoundError(f"idea file not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def load(cls, data_root: str, script_id: UUID | str) -> StoryIdea:
        return cls.load_json(idea_json_path(data_root, script_id))

    @classmethod
    def read_all(cls, data_root: str) -> list[StoryIdea]:
        """Load one StoryIdea per ``<script_id>/idea.json`` under data_root."""
        ideas: list[StoryIdea] = []
        for story_path in iter_story_dirs(data_root):
            idea_path = story_path / "idea.json"
            if idea_path.is_file():
                ideas.append(cls.load_json(idea_path))
        if not ideas:
            raise ValueError(f"no ideas found in {resolve_path(data_root)}")
        return ideas

    def save(self, data_root: str) -> None:
        out_dir = story_dir(data_root, self.script_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        with idea_json_path(data_root, self.script_id).open("w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2)


@dataclass
class SceneScript:
    """One scene written by stages 3–5 to ``<script_id>/<n>/script.json``."""

    script_id: UUID
    model: str
    scene: Scene
    scene_content: list[tuple[str, str]] = field(default_factory=list)
    image_prompt: list[ImagePrompt] | None = None

    @classmethod
    def parse_scene_dict(cls, data: Any) -> Scene:
        """Validate and normalize a scene outline dict from LLM output or JSON."""
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")

        missing = [name for name in _SCENE_FIELDS if name not in data]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")

        act = normalize_act(data["act"])
        scene_characters = coerce_character_list(data["characters"])
        scene_number = coerce_int_field(data["scene_number"], "scene_number")

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
        min_prompts: int = 0,
        max_prompts: int = 5,
        beat_count: int = 0,
        require_lines_used: bool = False,
    ) -> list[ImagePrompt] | None:
        """Validate null, a single prompt object, or a list of prompt objects."""
        if data is None:
            return None
        parse_kwargs = {
            "beat_count": beat_count,
            "require_lines_used": require_lines_used,
        }
        if isinstance(data, dict):
            prompts = [cls.parse_img_prompt_dict(data, **parse_kwargs)]
        elif isinstance(data, list):
            if not data:
                raise ValueError("image_prompt must be null or a non-empty array")
            prompts = [
                cls.parse_img_prompt_dict(item, **parse_kwargs) for item in data
            ]
        else:
            raise TypeError(
                f"image_prompt must be an object, array, or null, got {type(data).__name__}"
            )

        if min_prompts < 0:
            raise ValueError("min_prompts must be >= 0")
        if max_prompts < min_prompts:
            raise ValueError(
                f"max_prompts must be >= min_prompts ({max_prompts} < {min_prompts})"
            )
        if len(prompts) < min_prompts:
            raise ValueError(
                f"expected at least {min_prompts} image prompt(s) but parsed {len(prompts)}"
            )
        if len(prompts) > max_prompts:
            raise ValueError(
                f"expected at most {max_prompts} image prompt(s) but parsed {len(prompts)}"
            )
        return prompts

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "script_id": str(self.script_id),
            "model": self.model,
            "scene": dict(self.scene),
            "scene_content": _serialize_scene_content(self.scene_content),
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
    def read_for_story(cls, data_root: str, script_id: UUID | str) -> list[SceneScript]:
        story_path = story_dir(data_root, script_id)
        if not story_path.is_dir():
            raise FileNotFoundError(f"story directory not found: {story_path}")
        scripts: list[SceneScript] = []
        for scene_path in iter_scene_dirs(story_path):
            script_json = scene_path / "script.json"
            if script_json.is_file():
                scripts.append(cls.load_json(script_json))
        scripts.sort(key=lambda item: item.scene["scene_number"])
        return scripts

    @classmethod
    def read_all(cls, data_root: str) -> list[SceneScript]:
        """Load every ``<script_id>/<scene>/script.json`` under data_root."""
        scripts: list[SceneScript] = []
        for story_path in iter_story_dirs(data_root):
            if not (story_path / "idea.json").is_file():
                continue
            for scene_path in iter_scene_dirs(story_path):
                script_json = scene_path / "script.json"
                if script_json.is_file():
                    scripts.append(cls.load_json(script_json))
        if not scripts:
            raise ValueError(f"no scene scripts found in {resolve_path(data_root)}")
        scripts.sort(
            key=lambda item: (str(item.script_id), item.scene["scene_number"])
        )
        return scripts

    def save(self, data_root: str) -> None:
        scene_number = self.scene["scene_number"]
        out_dir = scene_dir(data_root, self.script_id, scene_number)
        out_dir.mkdir(parents=True, exist_ok=True)
        with scene_script_json_path(data_root, self.script_id, scene_number).open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(self.to_json(), fh, indent=2)


def clamp_scene_content_beats(
    content: list[tuple[str, str]],
    *,
    max_beats: int,
    head: int = 2,
    tail: int = 3,
) -> list[tuple[str, str]]:
    """Trim excess beats while preserving the opening and closing exchanges."""
    count = len(content)
    if count <= max_beats:
        return content

    head = min(head, max_beats)
    tail = min(tail, max_beats - head)
    middle_slots = max_beats - head - tail
    if tail:
        middle = content[head : count - tail]
        trimmed = content[:head] + middle[:middle_slots] + content[count - tail :]
    else:
        trimmed = content[:head] + content[head : head + middle_slots]
    return trimmed


def parse_scene_content(
    data: Any,
    *,
    min_beats: int | None = None,
    max_beats: int | None = None,
    clamp_over_max: bool = False,
) -> list[tuple[str, str]]:
    """Validate scene_content from LLM output or JSON."""
    if data is None:
        content: list[tuple[str, str]] = []
    elif not isinstance(data, list):
        raise TypeError("scene_content must be an array")
    else:
        content = []
        for index, item in enumerate(data):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                character = str(item[0]).strip()
                text = str(item[1])
                if not character:
                    raise ValueError(
                        f"scene_content[{index}]: character name must be non-empty"
                    )
                content.append((character, text))
                continue
            if isinstance(item, dict):
                character = str(
                    item.get("character") or item.get("speaker") or ""
                ).strip()
                text = str(item.get("text") or item.get("line") or "")
                if not character:
                    raise ValueError(
                        f"scene_content[{index}]: character name must be non-empty"
                    )
                content.append((character, text))
                continue
            raise ValueError(
                f"scene_content[{index}] must be a 2-element array or object with character/text"
            )

    if min_beats is not None and max_beats is not None:
        count = len(content)
        if count < min_beats:
            raise ValueError(
                f"expected {min_beats}-{max_beats} content pair(s) but parsed {count}"
            )
        if count > max_beats:
            if clamp_over_max:
                logging.getLogger(__name__).warning(
                    "Trimming scene_content from %s to %s beat(s) "
                    "(preserving opening and closing exchanges)",
                    count,
                    max_beats,
                )
                content = clamp_scene_content_beats(content, max_beats=max_beats)
            else:
                raise ValueError(
                    f"expected {min_beats}-{max_beats} content pair(s) but parsed {count}"
                )
    return content


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
