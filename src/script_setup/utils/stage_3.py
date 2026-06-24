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
    Script,
    resolve_path,
)


def run_stage(
    logger: logging.Logger,
    model,
    sampling_params,
    tokenizer,
    config: SceneOutlineConfig,
    enable_thinking: bool = False,
) -> list[Script]:
    logger.info("Beginning stage 3 (scene outline generation)")
    scripts = Script.read_all(config.script_path)
    logger.info("Loaded %s scripts from %s", len(scripts), config.script_path)
    logger.info(
        "Running vLLM generate for %s script(s) (script batch_size=%s)",
        len(scripts),
        config.batch_size,
    )
    start = time.perf_counter()

    if not scripts:
        raise ValueError("stage 3 received no scripts")

    missing_titles = [s for s in scripts if not s.raw_title]
    if missing_titles:
        missing_ids = ", ".join(str(script.script_id) for script in missing_titles)
        raise ValueError(
            f"stage 3 requires titles; missing for {len(missing_titles)} script(s): "
            f"{missing_ids}"
        )

    scripts = generate_scene_outlines(
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
    logger.info("Stage 3 total time: %.2fs", elapsed)
    return scripts


def generate_scene_outlines(
    logger: logging.Logger,
    model,
    scripts: list[Script],
    config: SceneOutlineConfig,
    sampling_params,
    tokenizer,
    enable_thinking: bool = False,
) -> list[Script]:
    if not scripts:
        return scripts

    num_scenes = config.num_scenes
    prompt_path = resolve_path(config.prompt_path)
    sys_prompt = load_prompt_md(str(prompt_path))
    if not sys_prompt:
        raise FileNotFoundError(f"Prompt file not found or empty: {prompt_path}")

    model_name = model.llm_engine.model_config.model
    max_attempts = 2
    pending_scripts = list(scripts)

    for attempt in range(1, max_attempts + 1):
        prompts = [
            _format_prompt(num_scenes, tokenizer, sys_prompt, script, enable_thinking)
            for script in pending_scripts
        ]
        logger.info(
            "Submitting %s scene outline prompt(s) for %s script(s) to vLLM "
            "(attempt %s/%s, script batch_size=%s)",
            len(prompts),
            len(pending_scripts),
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

        failed_scripts: list[Script] = []
        for script, request_output in zip(pending_scripts, raw_outputs):
            answer_text = strip_reasoning(request_output_text(request_output))
            try:
                script.script_scenes = parse_scenes_from_text(answer_text, num_scenes)
                script.model = model_name
            except ValueError as exc:
                logger.warning(
                    "Failed to parse scene outlines for script %s (attempt %s/%s): %s",
                    script.script_id,
                    attempt,
                    max_attempts,
                    exc,
                )
                logger.warning(
                    "Raw model output for script %s:\n%s",
                    script.script_id,
                    answer_text,
                )
                failed_scripts.append(script)

        if not failed_scripts:
            break
        if attempt >= max_attempts:
            failed_ids = ", ".join(str(script.script_id) for script in failed_scripts)
            raise ValueError(
                f"failed to parse scene outlines for script(s): {failed_ids}"
            )
        pending_scripts = failed_scripts

    return scripts


def _format_prompt(
    num_scenes: int,
    tokenizer,
    sys_prompt: str,
    script: Script,
    enable_thinking: bool,
) -> str:
    payload = script.prompt_payload()
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
            scene = Script.parse_scene_dict(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scene at index {scene_index}: {exc}") from exc
        if scene["scene_number"] != scene_index:
            raise ValueError(
                f"scene at index {scene_index}: scene_number {scene['scene_number']} does not match index"
            )
        scenes.append(scene)
    return scenes
