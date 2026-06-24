from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from utils.schema import GenerationConfig, InputOutputConfig, _resolve_path

logger = logging.getLogger(__name__)


def _coerce_prompt_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    return []


def format_prompt_tags(tags: list[str]) -> str:
    return ", ".join(tag for tag in tags if tag)


@dataclass
class Script:
    """Script metadata loaded from ``data/<script_id>/script.json``."""

    script_id: UUID
    script_scenes: list[dict[str, Any]] | None = None
    raw_title: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Script:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")

        script_id_raw = data.get("script_id")
        if not script_id_raw:
            idea = data.get("idea") or data.get("raw_idea") or {}
            if isinstance(idea, dict):
                script_id_raw = idea.get("idea_id")
        if not script_id_raw:
            raise ValueError("script.json missing script_id")

        raw_scenes = data.get("script_scenes")
        script_scenes: list[dict[str, Any]] | None = None
        if raw_scenes is not None:
            if not isinstance(raw_scenes, list):
                raise ValueError("script_scenes must be a list or null")
            script_scenes = []
            for index, item in enumerate(raw_scenes):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"script_scenes[{index}] is not valid JSON: {exc}"
                        ) from exc
                if not isinstance(item, dict):
                    raise ValueError(
                        f"script_scenes[{index}] must be an object, "
                        f"got {type(item).__name__}"
                    )
                script_scenes.append(item)

        raw_title = data.get("raw_title")
        if raw_title is not None:
            raw_title = str(raw_title).strip() or None

        return cls(
            script_id=UUID(str(script_id_raw)),
            script_scenes=script_scenes,
            raw_title=raw_title,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> Script:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"script file not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def load_for_script_id(
        cls,
        script_id: UUID | str,
        io_cfg: InputOutputConfig,
    ) -> Script:
        script_path = _resolve_path(io_cfg.script_path) / str(script_id) / "script.json"
        return cls.load_json(script_path)

    @classmethod
    def load_by_ids(
        cls,
        script_ids: list[str],
        io_cfg: InputOutputConfig,
    ) -> dict[str, Script]:
        scripts: dict[str, Script] = {}
        for script_id in script_ids:
            scripts[script_id] = cls.load_for_script_id(script_id, io_cfg)
        return scripts

    def scene_by_number(self, scene_number: int) -> dict[str, Any] | None:
        scenes = self.script_scenes or []
        for scene in scenes:
            if scene.get("scene_number") == scene_number:
                return scene
        if 0 <= scene_number < len(scenes):
            return scenes[scene_number]
        return None

    def scene_prompts(
        self,
        scene_number: int,
        gen_cfg: GenerationConfig,
    ) -> tuple[str, str]:
        """Positive/negative prompts for one scene from image_prompt, with config fallback."""
        default_positive = (gen_cfg.prompt or "").strip()
        default_negative = (gen_cfg.negative_prompt or "").strip()

        scene = self.scene_by_number(scene_number)
        if scene is None:
            return default_positive, default_negative

        image_prompt = scene.get("image_prompt")
        if not isinstance(image_prompt, dict):
            return default_positive, default_negative

        positive = format_prompt_tags(
            _coerce_prompt_tags(image_prompt.get("positive_prompt"))
        )
        negative = format_prompt_tags(
            _coerce_prompt_tags(image_prompt.get("negative_prompt"))
        )
        return positive or default_positive, negative or default_negative


def validate_scripts_for_video(
    logger: logging.Logger,
    scripts_by_id: dict[str, Script],
    scene_numbers_by_script: dict[str, list[int]],
    gen_cfg: GenerationConfig,
) -> None:
    """Ensure every scene has a positive prompt before loading a prompted video model."""
    errors: list[str] = []
    for script_id, scene_numbers in scene_numbers_by_script.items():
        script = scripts_by_id.get(script_id)
        if script is None:
            errors.append(f"{script_id}: script.json not loaded")
            continue
        for scene_number in scene_numbers:
            positive, _ = script.scene_prompts(scene_number, gen_cfg)
            if not positive:
                errors.append(
                    f"{script_id}: scene {scene_number} has no positive prompt "
                    "(set image_prompt.positive_prompt in script.json or "
                    "generation_config.prompt)"
                )

    if errors:
        raise ValueError(
            "video generation with prompted backends requires a positive prompt "
            "for every scene:\n" + "\n".join(f"  - {item}" for item in errors)
        )

    logger.info(
        "Validated prompts for %s script(s), %s scene(s)",
        len(scripts_by_id),
        sum(len(numbers) for numbers in scene_numbers_by_script.values()),
    )
