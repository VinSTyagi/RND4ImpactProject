from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID

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


__all__ = [
    "IMAGE_PROMPT_FIELDS",
    "ImagePrompt",
    "Scene",
    "SceneScript",
    "parse_scene_content",
    "resolve_path",
    "scene_payload",
]
