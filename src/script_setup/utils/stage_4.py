from __future__ import annotations

import json
import logging
import re
import time
from json import JSONDecodeError

from prompts.prompt_reader import load_prompt_md
from utils.batching import generate_prompts as run_llm_generations
from utils.llm_helper import (
    parse_json_array,
    parse_json_object,
    request_output_text,
    strip_markdown_fences,
    strip_reasoning,
)
from utils.schema import (
    Scene,
    SceneContentConfig,
    Script,
    idea_prompt_payload,
    parse_scene_content,
    resolve_path,
    scene_outline_payload,
)


def run_stage(
    logger: logging.Logger,
    model,
    sampling_params,
    tokenizer,
    config: SceneContentConfig,
    enable_thinking: bool = False,
) -> list[Script]:
    logger.info("Beginning stage 4 (scene script generation)")
    scripts = Script.read_all(config.script_path)
    logger.info("Loaded %s scripts from %s", len(scripts), config.script_path)
    logger.info(
        "Processing %s script(s) one at a time "
        "(scene batch_size=%s, beats=%s-%s)",
        len(scripts),
        config.batch_size,
        config.min_beats,
        config.max_beats,
    )
    start = time.perf_counter()

    if not scripts:
        raise ValueError("stage 4 received no scripts")

    missing_scenes = [x for x in scripts if not x.script_scenes]
    if missing_scenes:
        missing_ids = ", ".join(str(script.script_id) for script in missing_scenes)
        raise ValueError(
            f"stage 4 requires scene outlines from stage 3; missing for "
            f"{len(missing_scenes)} script(s): {missing_ids}"
        )

    if config.min_beats < 1:
        raise ValueError("scene_content_config.min_beats must be at least 1")
    if config.max_beats < config.min_beats:
        raise ValueError(
            "scene_content_config.max_beats must be >= min_beats "
            f"({config.max_beats} < {config.min_beats})"
        )

    scripts = generate_scene_content(
        logger,
        model,
        scripts,
        config,
        sampling_params,
        tokenizer,
        enable_thinking=enable_thinking,
    )

    for script in scripts:
        script.save(config.script_path)
    logger.info("Saved %s scripts to %s", len(scripts), config.script_path)

    elapsed = time.perf_counter() - start
    logger.info("Stage 4 total time: %.2fs", elapsed)
    return scripts


def generate_scene_content(
    logger: logging.Logger,
    model,
    scripts: list[Script],
    config: SceneContentConfig,
    sampling_params,
    tokenizer,
    enable_thinking: bool = False,
) -> list[Script]:
    prompt_path = resolve_path(config.prompt_path)
    system_prompt_template = load_prompt_md(str(prompt_path))
    if not system_prompt_template:
        raise ValueError(
            f"System prompt is empty or file path for the prompt is invalid: {config.prompt_path}"
        )
    system_prompt = _format_system_prompt(
        system_prompt_template,
        min_beats=config.min_beats,
        max_beats=config.max_beats,
    )

    model_name = model.llm_engine.model_config.model

    for script in scripts:
        scenes = script.script_scenes or []
        logger.info(
            "Script: %s — generating scene_content for %s scene(s) (batch_size=%s)",
            script.script_id,
            len(scenes),
            config.batch_size,
        )

        chat_prompts = [
            format_user_prompt(
                tokenizer,
                system_prompt,
                script,
                scene,
                min_beats=config.min_beats,
                max_beats=config.max_beats,
                enable_thinking=enable_thinking,
            )
            for scene in scenes
        ]

        scene_contents: list[list[tuple[str, str]] | None] = [None] * len(scenes)
        pending_indices = list(range(len(scenes)))
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            prompts = [chat_prompts[index] for index in pending_indices]
            raw_outputs = run_llm_generations(
                model,
                prompts,
                sampling_params,
                batch_size=config.batch_size if attempt == 1 else 1,
                logger=logger,
                label=f"stage 4 script {script.script_id}",
            )

            still_pending: list[int] = []
            for scene_index, request_output in zip(pending_indices, raw_outputs):
                scene = scenes[scene_index]
                try:
                    answer_text = strip_reasoning(request_output_text(request_output))
                    content = parse_content_from_text(answer_text)
                    validate_beat_count(
                        content,
                        min_beats=config.min_beats,
                        max_beats=config.max_beats,
                    )
                    scene_contents[scene_index] = content
                except ValueError as exc:
                    if attempt >= max_attempts:
                        raise ValueError(
                            f"failed to parse scene_content for script {script.script_id} "
                            f"scene {scene['scene_number']}: {exc}"
                        ) from exc
                    logger.warning(
                        "Failed to parse scene_content for script %s scene %s "
                        "(attempt %s/%s): %s",
                        script.script_id,
                        scene["scene_number"],
                        attempt,
                        max_attempts,
                        exc,
                    )
                    still_pending.append(scene_index)

            if not still_pending:
                break
            pending_indices = still_pending

        if any(content is None for content in scene_contents):
            raise ValueError(
                f"failed to generate scene_content for script {script.script_id}"
            )

        script.script_scenes = Script.merge_scene_content(
            scenes,
            [content for content in scene_contents if content is not None],
        )
        script.model = model_name

    return scripts


def _extract_scene_content_array_text(text: str) -> str | None:
    """Extract the JSON array assigned to scene_content, if present."""
    cleaned = strip_markdown_fences(text)
    match = re.search(
        r'"scene_content"\s*:\s*\[',
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        return None

    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return None


def _format_system_prompt(
    template: str,
    *,
    min_beats: int,
    max_beats: int,
) -> str:
    return (
        template.replace("{min_beats}", str(min_beats)).replace(
            "{max_beats}", str(max_beats)
        )
    )


def validate_beat_count(
    content: list[tuple[str, str]],
    *,
    min_beats: int,
    max_beats: int,
) -> None:
    count = len(content)
    if count < min_beats or count > max_beats:
        raise ValueError(
            f"expected {min_beats}-{max_beats} content pair(s) but parsed {count}"
        )


def parse_content_from_text(text: str) -> list[tuple[str, str]]:
    last_exc: ValueError | None = None

    for parser in (_parse_content_payload, _parse_content_array_fallback):
        try:
            return parser(text)
        except (JSONDecodeError, ValueError, TypeError) as exc:
            last_exc = ValueError(f"invalid JSON in model output: {exc}")

    assert last_exc is not None
    raise last_exc


def _parse_content_payload(text: str) -> list[tuple[str, str]]:
    payload = parse_json_object(text)

    if isinstance(payload, list):
        return parse_scene_content(payload)

    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object with scene_content")

    if "scene_content" not in payload:
        raise ValueError("missing field: scene_content")

    return parse_scene_content(payload["scene_content"])


def _parse_content_array_fallback(text: str) -> list[tuple[str, str]]:
    array_text = _extract_scene_content_array_text(text)
    if not array_text:
        raise ValueError("no scene_content array found in model output")

    payload = parse_json_array(array_text)
    return parse_scene_content(payload)


def format_user_prompt(
    tokenizer,
    system_prompt: str,
    script: Script,
    scene: Scene,
    *,
    min_beats: int,
    max_beats: int,
    enable_thinking: bool = False,
) -> str:
    if not script.raw_title:
        raise ValueError(f"script {script.script_id} missing raw_title")

    story_payload = idea_prompt_payload(script.idea)
    story_payload["title"] = script.raw_title

    user_payload = {
        "story": story_payload,
        "scene": scene_outline_payload(scene),
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Here is the story and scene outline. Expand the outline into a wide, "
                "full scene script with complex dialogue and subtext.\n"
                "Use the story's premise, hook, tone, and theme to deepen stakes. "
                "Structure: opening beat → escalating conflict → turn → ends_on. "
                "Return a single JSON object with one field, scene_content: an array "
                f"of {min_beats}–{max_beats} [character, line] pairs.\n"
                f"{json.dumps(user_payload, ensure_ascii=False)}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
