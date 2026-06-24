from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

from PIL import Image

import yaml

from utils.image_prompt import coerce_prompt_tags, normalize_image_prompt_fields

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
    scene_content: list[tuple[str, str]]
    act: str
    setting: str
    characters: list[str]
    summary: str
    conflict: str
    emotional_beat: str
    character_change: str
    ends_on: str
    image_prompt: list[ImagePrompt] | None


def _serialize_scene_content(
    content: list[tuple[str, str]],
) -> list[list[str]]:
    return [[character, text] for character, text in content]


def scene_outline_payload(scene: Scene) -> dict[str, Any]:
    """Outline fields only, for stage 4 prompts."""
    return {name: scene[name] for name in _SCENE_FIELDS}


def scene_payload(scene: Scene) -> dict[str, Any]:
    """Outline plus scene_content, for stage 5 image-prompt prompts."""
    payload = scene_outline_payload(scene)
    payload["scene_content"] = _serialize_scene_content(scene.get("scene_content") or [])
    return payload


def parse_scene_content(data: Any) -> list[tuple[str, str]]:
    """Validate scene_content from LLM output or JSON."""
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
        if isinstance(item, dict):
            character = str(item.get("character") or item.get("speaker") or "").strip()
            text = str(item.get("text") or item.get("line") or "")
            if not character:
                raise ValueError(f"scene_content[{index}]: character name must be non-empty")
            content.append((character, text))
            continue
        raise ValueError(
            f"scene_content[{index}] must be a 2-element array or object with character/text"
        )
    return content


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

        scene_content = parse_scene_content(data.get("scene_content"))

        image_prompt_data = data.get("image_prompt")
        scene_image_prompt = cls.parse_img_prompt_list(image_prompt_data)

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
            "scene_content": scene_content,
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
    def parse_img_prompt_list(
        cls,
        data: Any,
        *,
        min_prompts: int = 0,
        max_prompts: int = 5,
    ) -> list[ImagePrompt] | None:
        """Validate null, a single prompt object, or a list of prompt objects."""
        if data is None:
            return None
        if isinstance(data, dict):
            prompts = [cls.parse_img_prompt_dict(data)]
        elif isinstance(data, list):
            if not data:
                raise ValueError("image_prompt must be null or a non-empty array")
            prompts = [cls.parse_img_prompt_dict(item) for item in data]
        else:
            raise TypeError(f"image_prompt must be an object, array, or null, got {type(data).__name__}")

        if min_prompts and len(prompts) < min_prompts:
            raise ValueError(
                f"expected at least {min_prompts} image prompt(s) but parsed {len(prompts)}"
            )
        if max_prompts and len(prompts) > max_prompts:
            raise ValueError(
                f"expected at most {max_prompts} image prompt(s) but parsed {len(prompts)}"
            )
        return prompts

    @classmethod
    def attach_scene_image_prompts(
        cls,
        scenes: list[Scene],
        prompts_by_scene: list[list[ImagePrompt]],
        *,
        min_prompts: int = 1,
        max_prompts: int = 5,
    ) -> list[Scene]:
        """Return scenes with image_prompt set from per-scene prompt lists."""
        if len(scenes) != len(prompts_by_scene):
            raise ValueError(
                f"expected {len(scenes)} scene prompt list(s) but received "
                f"{len(prompts_by_scene)}"
            )
        updated: list[Scene] = []
        for scene_index, (scene, prompts) in enumerate(zip(scenes, prompts_by_scene)):
            count = len(prompts)
            if count < min_prompts or count > max_prompts:
                raise ValueError(
                    f"scene {scene_index}: expected {min_prompts}-{max_prompts} "
                    f"image prompt(s) but received {count}"
                )
            updated.append({**scene, "image_prompt": prompts})
        return updated

    @classmethod
    def merge_scene_content(
        cls,
        scenes: list[Scene],
        scene_contents: list[list[tuple[str, str]]],
    ) -> list[Scene]:
        """Return scenes with scene_content filled from a parallel content list."""
        if len(scenes) != len(scene_contents):
            raise ValueError(
                f"expected {len(scenes)} scene_content list(s) but received "
                f"{len(scene_contents)}"
            )
        return [
            {**scene, "scene_content": content}
            for scene, content in zip(scenes, scene_contents)
        ]

    def prompt_payload(self) -> dict[str, str]:
        """Story fields plus title, for stage 3 prompts."""
        if not self.raw_title:
            raise ValueError(f"script {self.script_id} missing raw_title")
        payload = idea_prompt_payload(self.idea)
        payload["title"] = self.raw_title
        return payload

    def to_json(self) -> dict[str, Any]:
        scenes_out: list[dict[str, Any]] | None = None
        if self.script_scenes is not None:
            scenes_out = []
            for scene in self.script_scenes:
                scene_dict: dict[str, Any] = dict(scene)
                scene_dict["scene_content"] = _serialize_scene_content(
                    scene.get("scene_content") or []
                )
                scenes_out.append(scene_dict)
        return {
            "idea": self.idea,
            "script_id": str(self.script_id),
            "raw_title": self.raw_title,
            "script_scenes": scenes_out,
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


@dataclass
class VLLMModelConfig:
    model_path: str = "Qwen/Qwen3-4B-AWQ"
    quantization: str = "awq"
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
    language_model_only: bool = False
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 8192


@dataclass
class IdeaConfig:
    num_ideas: int = 5
    prompt_path: str = "script_setup/prompts/stage_1.md"
    output_path: str = "data/"


@dataclass
class TitleConfig:
    num_words: int = 10
    prompt_path: str = "script_setup/prompts/stage_2.md"
    script_path: str = "data/"


@dataclass
class SceneOutlineConfig:
    num_scenes: int = 5
    prompt_path: str = "script_setup/prompts/stage_3.md"
    script_path: str = "data/"


@dataclass
class SceneContentConfig:
    prompt_path: str = "script_setup/prompts/stage_4.md"
    script_path: str = "data/"


@dataclass
class ImagePromptConfig:
    min_prompts: int = 1
    max_prompts: int = 5
    prompt_path: str = "script_setup/prompts/stage_5.md"
    script_path: str = "data/"


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


def load_config(path: str) -> PipelineConfig:
    data = _load_yaml(path)
    scene_outline_section = _section(data, "scene_outline_config") or _section(
        data, "scene_config"
    )
    return PipelineConfig(
        global_vllm_config=VLLMModelConfig(**_section(data, "global_vllm_config")),
        idea_config=IdeaConfig(**_section(data, "idea_config")),
        title_config=TitleConfig(**_section(data, "title_config")),
        scene_outline_config=SceneOutlineConfig(**scene_outline_section),
        scene_content_config=SceneContentConfig(
            **_section(data, "scene_content_config")
        ),
        image_config=ImagePromptConfig(**_section(data, "image_config")),
    )
