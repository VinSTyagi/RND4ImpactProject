from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID

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




class Scene(TypedDict, total=False):
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
    """Parse scene_content from script.json."""
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError("scene_content must be an array")

    content: list[tuple[str, str]] = []
    for index, item in enumerate(data):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            character = str(item[0]).strip()
            text = str(item[1])
        elif isinstance(item, dict):
            character = str(
                item.get("character") or item.get("speaker") or ""
            ).strip()
            text = str(item.get("text") or item.get("line") or "")
        else:
            raise ValueError(
                f"scene_content[{index}] must be a 2-element array or object "
                "with character/text"
            )
        if not character:
            raise ValueError(
                f"scene_content[{index}]: character name must be non-empty"
            )
        content.append((character, text))
    return content


def _story_dir(data_root: str, script_id: UUID | str) -> Path:
    return resolve_path(data_root) / str(script_id)


def idea_json_path(data_root: str, script_id: UUID | str) -> Path:
    return _story_dir(data_root, script_id) / "idea.json"


def coerce_character_descriptions(value: Any) -> dict[str, str]:
    """Normalize story-bible character map from idea.json."""
    if not isinstance(value, dict):
        raise ValueError(
            "characters must be a non-empty object mapping names to descriptions"
        )
    profiles: dict[str, str] = {}
    for key, desc in value.items():
        name = str(key).strip()
        description = str(desc).strip()
        if not name:
            raise ValueError("characters keys must be non-empty names")
        if not description:
            raise ValueError(f"characters[{name!r}] description must be non-empty")
        profiles[name] = description
    if not profiles:
        raise ValueError("characters must contain at least one entry")
    return profiles


_IDEA_STRING_FIELDS = (
    "genre",
    "setting",
    "premise",
    "hook",
    "tone",
    "theme",
)


@dataclass
class StoryIdea:
    """Story bible from ``<script_id>/idea.json``; used for TTS voice profiles."""

    genre: str
    setting: str
    premise: str
    characters: dict[str, str]
    hook: str
    tone: str
    theme: str
    script_id: UUID
    model: str = ""
    title: str | None = None

    @classmethod
    def _idea_fields_from_dict(cls, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        missing = [
            name
            for name in _IDEA_STRING_FIELDS
            if name not in data or not str(data[name]).strip()
        ]
        if "characters" not in data:
            missing.append("characters")
        if missing:
            raise ValueError(f"missing or empty fields: {', '.join(missing)}")
        fields_out = {name: str(data[name]).strip() for name in _IDEA_STRING_FIELDS}
        fields_out["characters"] = coerce_character_descriptions(data["characters"])
        return fields_out

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


def _iter_story_dirs(data_root: str) -> list[Path]:
    base = resolve_path(data_root)
    if not base.is_dir():
        return []
    return sorted(
        (entry for entry in base.iterdir() if entry.is_dir()),
        key=lambda path: path.name,
    )


def _iter_scene_dirs(story_path: Path) -> list[Path]:
    return sorted(
        (
            entry
            for entry in story_path.iterdir()
            if entry.is_dir() and entry.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )


@dataclass
class SceneScript:
    """One scene loaded from ``<script_id>/<n>/script.json`` for TTS."""

    script_id: UUID
    model: str
    scene: Scene
    scene_content: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneScript:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")

        script_id_raw = data.get("script_id")
        if not script_id_raw:
            raise ValueError("missing script_id")

        scene_data = data.get("scene")
        if not isinstance(scene_data, dict):
            raise ValueError("missing scene")

        model = str(data.get("model") or "").strip()
        scene_content = parse_scene_content(data.get("scene_content"))

        return cls(
            script_id=UUID(str(script_id_raw)),
            model=model,
            scene=scene_data,  # type: ignore[arg-type]
            scene_content=scene_content,
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
        story_path = _story_dir(data_root, script_id)
        if not story_path.is_dir():
            raise FileNotFoundError(f"story directory not found: {story_path}")
        scripts: list[SceneScript] = []
        for scene_path in _iter_scene_dirs(story_path):
            script_json = scene_path / "script.json"
            if script_json.is_file():
                scripts.append(cls.load_json(script_json))
        scripts.sort(key=lambda item: int(item.scene.get("scene_number", 0)))
        return scripts

    @classmethod
    def read_all(cls, data_root: str) -> list[SceneScript]:
        """Load every ``<script_id>/<scene>/script.json`` under data_root."""
        scripts: list[SceneScript] = []
        for story_path in _iter_story_dirs(data_root):
            for scene_path in _iter_scene_dirs(story_path):
                script_json = scene_path / "script.json"
                if script_json.is_file():
                    scripts.append(cls.load_json(script_json))
        if not scripts:
            raise ValueError(f"no scene scripts found in {resolve_path(data_root)}")
        scripts.sort(
            key=lambda item: (str(item.script_id), int(item.scene.get("scene_number", 0)))
        )
        return scripts


__all__ = [
    "Scene",
    "SceneScript",
    "StoryIdea",
    "coerce_character_descriptions",
    "idea_json_path",
    "parse_scene_content",
    "resolve_path",
]
