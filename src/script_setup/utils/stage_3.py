from __future__ import annotations

import json
import logging
import time

from prompts.prompt_reader import load_prompt_md
from utils.batching import generate_prompts as run_llm_generations
from utils.llm_helper import (
    parse_json_array,
    request_output_text,
    strip_reasoning,
)
from utils.schema import (
    Scene,
    SceneOutlineConfig,
    SceneScript,
    StoryIdea,
    resolve_path,
)


def run_stage(
    logger: logging.Logger,
    model,
    sampling_params,
    tokenizer,
    config: SceneOutlineConfig,
    enable_thinking: bool = False,
) -> list[SceneScript]:
    logger.info("Beginning stage 3 (scene outline generation)")
    ideas = StoryIdea.read_all(config.script_path)
    logger.info("Loaded %s ideas from %s", len(ideas), config.script_path)
    logger.info(
        "Running vLLM generate for %s story idea(s) (batch_size=%s)",
        len(ideas),
        config.batch_size,
    )
    start = time.perf_counter()

    if not ideas:
        raise ValueError("stage 3 received no ideas")

    missing_titles = [idea for idea in ideas if not idea.title]
    if missing_titles:
        missing_ids = ", ".join(str(idea.script_id) for idea in missing_titles)
        raise ValueError(
            f"stage 3 requires titles; missing for {len(missing_titles)} story idea(s): "
            f"{missing_ids}"
        )

    scene_scripts = generate_scene_outlines(
        logger,
        model,
        ideas,
        config,
        sampling_params,
        tokenizer,
        enable_thinking=enable_thinking,
    )

    for scene_script in scene_scripts:
        scene_script.save(config.script_path)
    logger.info(
        "Saved %s scene script(s) under %s",
        len(scene_scripts),
        config.script_path,
    )

    elapsed = time.perf_counter() - start
    logger.info("Stage 3 total time: %.2fs", elapsed)
    return scene_scripts


def generate_scene_outlines(
    logger: logging.Logger,
    model,
    ideas: list[StoryIdea],
    config: SceneOutlineConfig,
    sampling_params,
    tokenizer,
    enable_thinking: bool = False,
) -> list[SceneScript]:
    if not ideas:
        return []

    num_scenes = config.num_scenes
    prompt_path = resolve_path(config.prompt_path)
    sys_prompt = load_prompt_md(str(prompt_path))
    if not sys_prompt:
        raise FileNotFoundError(f"Prompt file not found or empty: {prompt_path}")

    model_name = model.llm_engine.model_config.model
    max_attempts = 2
    pending_ideas = list(ideas)
    scene_scripts: list[SceneScript] = []

    for attempt in range(1, max_attempts + 1):
        prompts = [
            _format_prompt(num_scenes, tokenizer, sys_prompt, idea, enable_thinking)
            for idea in pending_ideas
        ]
        logger.info(
            "Submitting %s scene outline prompt(s) for %s story idea(s) to vLLM "
            "(attempt %s/%s, batch_size=%s)",
            len(prompts),
            len(pending_ideas),
            attempt,
            max_attempts,
            config.batch_size,
        )
        generate_start = time.perf_counter()
        raw_outputs = run_llm_generations(
            model,
            prompts,
            sampling_params,
            batch_size=config.batch_size,
            logger=logger,
            label="stage 3",
        )
        generate_elapsed = time.perf_counter() - generate_start
        logger.info("vLLM generate completed in %.2fs", generate_elapsed)

        failed_ideas: list[StoryIdea] = []
        for idea, request_output in zip(pending_ideas, raw_outputs):
            answer_text = strip_reasoning(request_output_text(request_output))
            try:
                scenes = parse_scenes_from_text(answer_text, num_scenes)
                for scene in scenes:
                    scene_scripts.append(
                        SceneScript(
                            script_id=idea.script_id,
                            model=model_name,
                            scene=scene,
                        )
                    )
            except ValueError as exc:
                logger.warning(
                    "Failed to parse scene outlines for story %s (attempt %s/%s): %s",
                    idea.script_id,
                    attempt,
                    max_attempts,
                    exc,
                )
                logger.warning(
                    "Raw model output for story %s:\n%s",
                    idea.script_id,
                    answer_text,
                )
                failed_ideas.append(idea)

        if not failed_ideas:
            break
        if attempt >= max_attempts:
            failed_ids = ", ".join(str(idea.script_id) for idea in failed_ideas)
            raise ValueError(
                f"failed to parse scene outlines for story idea(s): {failed_ids}"
            )
        pending_ideas = failed_ideas
        scene_scripts = []

    return scene_scripts


def _format_prompt(
    num_scenes: int,
    tokenizer,
    sys_prompt: str,
    idea: StoryIdea,
    enable_thinking: bool,
) -> str:
    payload = idea.prompt_payload()
    user_prompt = (
        f"Here is a story item JSON object. Produce exactly {num_scenes} scene outline objects as a flat JSON array ordered by scene_number starting at 0.\n"
        "Use thinking/reasoning if helpful, then output ONLY a complete JSON array that starts with `[` and ends with `]`.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def parse_scenes_from_text(text: str, num_scenes: int) -> list[Scene]:
    """Parse LLM text into a validated flat scene outline list."""
    try:
        payload = parse_json_array(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in model output: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("expected a JSON array of scene objects")
    if len(payload) != num_scenes:
        raise ValueError(f"expected {num_scenes} scene(s) but parsed {len(payload)}")

    scenes: list[Scene] = []
    for scene_index, item in enumerate(payload):
        try:
            scene = SceneScript.parse_scene_dict(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scene at index {scene_index}: {exc}") from exc
        if scene["scene_number"] != scene_index:
            raise ValueError(
                f"scene at index {scene_index}: scene_number {scene['scene_number']} does not match index"
            )
        scenes.append(scene)
    return scenes
