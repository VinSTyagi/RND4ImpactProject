from __future__ import annotations

import json
import logging
import time
from json import JSONDecodeError

from prompts.prompt_reader import load_prompt_md
from utils.batching import generate_prompts as run_llm_generations
from utils.llm_helper import (
    format_system_prompt,
    parse_json_array,
    request_output_text,
    strip_reasoning,
)
from utils.config import ImagePromptConfig
from utils.schema import (
    SceneScript,
    StoryIdea,
    cast_visual_context,
    resolve_path,
    scene_image_prompt_payload,
)


def run_stage(
    logger: logging.Logger,
    model,
    sampling_params,
    tokenizer,
    config: ImagePromptConfig,
    enable_thinking: bool = False,
) -> list[SceneScript]:
    logger.info("Beginning stage 5 (SDXL image prompt generation)")
    ideas = {
        str(idea.script_id): idea for idea in StoryIdea.read_all(config.script_path)
    }
    scene_scripts = SceneScript.read_all(config.script_path)
    logger.info(
        "Loaded %s scene script(s) from %s",
        len(scene_scripts),
        config.script_path,
    )
    logger.info(
        "Processing %s scene script(s) "
        "(scene batch_size=%s, image prompts=%s-%s)",
        len(scene_scripts),
        config.batch_size,
        config.min_prompts,
        config.max_prompts,
    )
    start = time.perf_counter()

    if not scene_scripts:
        raise ValueError("stage 5 received no scene scripts")

    if config.min_prompts < 1:
        raise ValueError("image_config.min_prompts must be at least 1")
    if config.max_prompts < config.min_prompts:
        raise ValueError(
            "image_config.max_prompts must be >= min_prompts "
            f"({config.max_prompts} < {config.min_prompts})"
        )

    scene_scripts = generate_prompts(
        logger,
        model,
        ideas,
        scene_scripts,
        config,
        sampling_params,
        tokenizer,
        enable_thinking=enable_thinking,
    )

    for scene_script in scene_scripts:
        scene_script.save(config.script_path)
    logger.info("Saved %s scene script(s) to %s", len(scene_scripts), config.script_path)

    elapsed = time.perf_counter() - start
    logger.info("Stage 5 total time: %.2fs", elapsed)
    return scene_scripts


def generate_prompts(
    logger: logging.Logger,
    model,
    ideas: dict[str, StoryIdea],
    scene_scripts: list[SceneScript],
    config: ImagePromptConfig,
    sampling_params,
    tokenizer,
    enable_thinking: bool = False,
) -> list[SceneScript]:
    prompt_path = resolve_path(config.prompt_path)
    system_prompt_template = load_prompt_md(str(prompt_path))
    if not system_prompt_template:
        raise ValueError(
            f"System prompt is empty or file path for the prompt is invalid: {config.prompt_path}"
        )
    system_prompt = format_system_prompt(
        system_prompt_template,
        min_prompts=str(config.min_prompts),
        max_prompts=str(config.max_prompts),
    )

    model_name = model.llm_engine.model_config.model
    min_prompts = config.min_prompts
    max_prompts = config.max_prompts

    for scene_script in scene_scripts:
        idea = ideas.get(str(scene_script.script_id))
        if idea is None:
            raise ValueError(
                f"missing idea.json for story {scene_script.script_id} "
                "(run stages 1–2 first)"
            )
        scene = scene_script.scene
        beat_count = len(scene_script.scene_content)
        logger.info(
            "Story %s scene %s — image prompt generation (batch_size=%s)",
            scene_script.script_id,
            scene["scene_number"],
            config.batch_size,
        )
        chat_prompts = format_user_prompts(
            tokenizer,
            system_prompt,
            idea,
            scene_script,
            min_prompts=min_prompts,
            max_prompts=max_prompts,
            enable_thinking=enable_thinking,
        )

        max_attempts = 2
        prompts_by_scene = None

        for attempt in range(1, max_attempts + 1):
            generate_start = time.perf_counter()
            raw_outputs = run_llm_generations(
                model,
                chat_prompts,
                sampling_params,
                batch_size=config.batch_size if attempt == 1 else 1,
                logger=logger,
                label=(
                    f"stage 5 story {scene_script.script_id} "
                    f"scene {scene['scene_number']}"
                ),
            )
            generate_elapsed = time.perf_counter() - generate_start
            logger.info("vLLM generate completed in %.2fs", generate_elapsed)

            try:
                answer_text = strip_reasoning(request_output_text(raw_outputs[0]))
                prompts_by_scene = parse_prompts_from_text(
                    answer_text,
                    min_prompts=min_prompts,
                    max_prompts=max_prompts,
                    beat_count=beat_count,
                )
                break
            except ValueError as exc:
                if attempt >= max_attempts:
                    raise ValueError(
                        f"failed to parse image prompts for story {scene_script.script_id} "
                        f"scene {scene['scene_number']}: {exc}"
                    ) from exc
                logger.warning(
                    "Failed to parse image prompts for story %s scene %s "
                    "(attempt %s/%s): %s",
                    scene_script.script_id,
                    scene["scene_number"],
                    attempt,
                    max_attempts,
                    exc,
                )

        if prompts_by_scene is None:
            raise ValueError(
                f"failed to generate image prompts for story {scene_script.script_id} "
                f"scene {scene['scene_number']}"
            )

        scene_script.image_prompt = prompts_by_scene
        scene_script.model = model_name

    return scene_scripts


def parse_prompts_from_text(
    text: str,
    *,
    min_prompts: int,
    max_prompts: int,
    beat_count: int,
):
    try:
        payload = parse_json_array(text)
    except JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in model output: {exc}") from exc

    prompts = SceneScript.parse_img_prompt_list(
        payload,
        min_prompts=min_prompts,
        max_prompts=max_prompts,
        beat_count=beat_count,
        require_lines_used=True,
    )
    if prompts is None:
        raise ValueError("expected a JSON array of image prompt objects")
    return prompts


def format_user_prompts(
    tokenizer,
    system_prompt: str,
    idea: StoryIdea,
    scene_script: SceneScript,
    *,
    min_prompts: int,
    max_prompts: int,
    enable_thinking: bool = False,
) -> list[str]:
    cast_descriptions = cast_visual_context(idea)
    scene_payload = scene_image_prompt_payload(
        scene_script.scene,
        scene_script.scene_content,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Here is a scene JSON object. Produce a JSON array with no fewer than "
                f"{min_prompts} and no more than {max_prompts} Stable Diffusion XL "
                "prompt objects (inclusive range).\n"
                f"{json.dumps({'cast_descriptions': cast_descriptions, 'scene': scene_payload}, ensure_ascii=False)}"
            ),
        },
    ]
    return [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    ]
