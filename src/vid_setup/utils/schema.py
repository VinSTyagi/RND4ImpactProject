from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from utils.config import InputOutputConfig


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
class SceneScript:
    """One scene loaded from ``<script_id>/<n>/script.json``."""

    script_id: UUID
    model: str = ""
    scene: dict[str, Any] | None = None
    image_prompt: list[dict[str, Any]] | dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneScript:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")

        script_id_raw = data.get("script_id")
        if not script_id_raw:
            raise ValueError("script.json missing script_id")

        scene_data = data.get("scene")
        if scene_data is None:
            raise ValueError("script.json missing scene")
        if not isinstance(scene_data, dict):
            raise ValueError("scene must be an object")

        model = str(data.get("model") or "").strip()
        image_prompt = data.get("image_prompt")
        if image_prompt is None:
            image_prompt = scene_data.get("image_prompt")
        return cls(
            script_id=UUID(str(script_id_raw)),
            model=model,
            scene=scene_data,
            image_prompt=image_prompt,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> SceneScript:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"scene script file not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def load_for_scene(
        cls,
        script_id: UUID | str,
        scene_number: int,
        io_cfg: InputOutputConfig,
    ) -> SceneScript:
        from utils.config import _resolve_path

        script_path = (
            _resolve_path(io_cfg.script_path)
            / str(script_id)
            / str(scene_number)
            / "script.json"
        )
        return cls.load_json(script_path)

    @classmethod
    def read_for_story(
        cls,
        script_id: UUID | str,
        io_cfg: InputOutputConfig,
    ) -> list[SceneScript]:
        from utils.config import _resolve_path

        story_path = _resolve_path(io_cfg.script_path) / str(script_id)
        if not story_path.is_dir():
            raise FileNotFoundError(f"story directory not found: {story_path}")
        scripts: list[SceneScript] = []
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
        scripts.sort(key=lambda item: item.scene_number())
        return scripts

    @classmethod
    def read_all(cls, data_root: str) -> list[SceneScript]:
        from utils.config import _resolve_path

        base = _resolve_path(data_root)
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
        return scripts

    @classmethod
    def load_by_ids(
        cls,
        script_ids: list[str],
        io_cfg: InputOutputConfig,
    ) -> dict[str, list[SceneScript]]:
        scripts: dict[str, list[SceneScript]] = {}
        for script_id in script_ids:
            scripts[script_id] = cls.read_for_story(script_id, io_cfg)
        return scripts

    def scene_number(self) -> int:
        scene = self.scene or {}
        try:
            return int(scene.get("scene_number", 0))
        except (TypeError, ValueError):
            return 0

    def scene_prompts(
        self,
        scene_number: int,
        prompt_number: int,
    ) -> tuple[str, str]:
        scene = self.scene
        if scene is None or scene.get("scene_number") != scene_number:
            return "", ""

        image_prompt_data = self.image_prompt
        if image_prompt_data is None:
            image_prompt_data = scene.get("image_prompt")
        if image_prompt_data is None:
            return "", ""

        if isinstance(image_prompt_data, dict):
            prompts = [image_prompt_data]
        elif isinstance(image_prompt_data, list):
            prompts = image_prompt_data
        else:
            return "", ""

        if prompt_number < 0 or prompt_number >= len(prompts):
            return "", ""

        image_prompt = prompts[prompt_number]
        if not isinstance(image_prompt, dict):
            return "", ""

        positive = format_prompt_tags(
            _coerce_prompt_tags(image_prompt.get("positive_prompt"))
        )
        negative = format_prompt_tags(
            _coerce_prompt_tags(image_prompt.get("negative_prompt"))
        )
        return positive, negative


Script = SceneScript


def validate_scripts_for_video(
    log: logging.Logger,
    scene_scripts_by_id: dict[str, list[SceneScript]],
    prompt_counts: dict[str, int],
) -> None:
    errors: list[str] = []
    for script_id, total_prompts in prompt_counts.items():
        scene_scripts = scene_scripts_by_id.get(script_id)
        if not scene_scripts:
            errors.append(f"{script_id}: no per-scene script.json loaded")
            continue
        checked = 0
        for scene_script in sorted(scene_scripts, key=lambda item: item.scene_number()):
            scene = scene_script.scene or {}
            scene_number = scene_script.scene_number()
            image_prompt_data = scene_script.image_prompt
            if image_prompt_data is None:
                image_prompt_data = scene.get("image_prompt")
            if isinstance(image_prompt_data, dict):
                prompt_items = [image_prompt_data]
            elif isinstance(image_prompt_data, list):
                prompt_items = image_prompt_data
            else:
                prompt_items = []
            for prompt_number in range(len(prompt_items)):
                positive, _ = scene_script.scene_prompts(scene_number, prompt_number)
                if not positive:
                    errors.append(
                        f"{script_id}: scene {scene_number} prompt {prompt_number} "
                        "has no positive prompt "
                        "(set image_prompt.positive_prompt in "
                        "data/<script_id>/<scene>/script.json)"
                    )
                checked += 1
        if checked != total_prompts:
            errors.append(
                f"{script_id}: expected {total_prompts} prompt(s) from paths "
                f"but per-scene script.json defines {checked}"
            )

    if errors:
        raise ValueError(
            "video generation with prompted backends requires a positive prompt "
            "for every scene prompt:\n" + "\n".join(f"  - {item}" for item in errors)
        )

    log.info(
        "Validated prompts for %s script(s), %s scene prompt(s)",
        len(scene_scripts_by_id),
        sum(prompt_counts.values()),
    )
