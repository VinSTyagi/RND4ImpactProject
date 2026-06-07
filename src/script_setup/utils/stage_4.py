from __future__ import annotations

import json
import logging
import time
from json import JSONDecodeError
from typing import Any

from prompts.prompt_reader import load_prompt_md
from utils.llm_helper import (
    extract_json_array_text,
    request_output_text,
    strip_reasoning,
)
from utils.schema import (
    ImagePrompt,
    ImagePromptConfig,
    Scene,
    Script,
    resolve_path,
    scene_payload,
)


def run_stage(
    logger: logging.Logger,
    model,
    sampling_params,
    tokenizer,
    scripts: list[Script],
    config: ImagePromptConfig,
    enable_thinking: bool = False,
) -> list[Script]:
    logger.info("Beginning stage 4 (SDXL image prompt generation)")
    logger.info("Running vLLM generate for %s scripts", len(scripts))
    start = time.perf_counter()

    if not scripts:
        raise ValueError("stage 4 received no scripts")

    missing_scenes = [x for x in scripts if not x.script_scenes]
    if missing_scenes:
        missing_ids = ", ".join(str(script.script_id) for script in missing_scenes)
        raise ValueError(
            f"stage 4 requires all scripts to have at least one scene; missing for {len(missing_scenes)} script(s): "
            f"{missing_ids}"
        )

    scripts = generate_prompts(
        logger,
        model,
        scripts,
        config,
        sampling_params,
        tokenizer,
        enable_thinking=enable_thinking,
    )

    elapsed = time.perf_counter() - start
    logger.info("Stage 4 total time: %.2fs", elapsed)
    return scripts


def generate_prompts(
    logger: logging.Logger,
    model,
    scripts: list[Script],
    config: ImagePromptConfig,
    sampling_params,
    tokenizer,
    enable_thinking: bool = False,
) -> list[Script]:
    prompt_path = resolve_path(config.prompt_path)
    system_prompt = load_prompt_md(str(prompt_path))
    if not system_prompt:
        raise ValueError(
            f"System prompt is empty or file path for the prompt is invalid: {config.prompt_path}"
        )

    model_name = model.llm_engine.model_config.model

    for script in scripts:
        scenes = script.script_scenes
        logger.info(
            "Script: %s — submitting %s scene(s) for image prompt generation",
            script.script_id,
            len(scenes),
        )
        chat_prompts = format_user_prompts(
            tokenizer,
            system_prompt,
            scenes,
            enable_thinking=enable_thinking,
        )
        generate_start = time.perf_counter()
        raw_outputs = model.generate(chat_prompts, sampling_params)
        generate_elapsed = time.perf_counter() - generate_start
        logger.info("vLLM generate completed in %.2fs", generate_elapsed)

        if len(raw_outputs) != len(scenes):
            raise ValueError(
                f"expected {len(scenes)} vLLM output(s) for script {script.script_id} "
                f"but received {len(raw_outputs)}"
            )

        image_prompts: list[ImagePrompt] = []
        for i, output in enumerate(raw_outputs):
            try:
                answer_text = strip_reasoning(request_output_text(output))
                image_prompts.extend(parse_prompts_from_text(answer_text, num_scenes=1))
            except ValueError as exc:
                raise ValueError(
                    f"failed to parse image prompt for script {script.script_id} "
                    f"scene {i}: {exc}"
                ) from exc

        script.script_scenes = Script.attach_image_prompts(scenes, image_prompts)
        script.model = model_name

    return scripts


def parse_prompts_from_text(text: str, num_scenes: int) -> list[ImagePrompt]:
    try:
        payload = json.loads(extract_json_array_text(text))
    except JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in model output: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("expected a JSON array of image prompt objects")

    prompts: list[ImagePrompt] = []
    for i, item in enumerate(payload):
        try:
            image_prompt = Script.parse_img_prompt_dict(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"image prompt at index {i}: {exc}") from exc

        prompts.append(image_prompt)

    if len(prompts) != num_scenes:
        raise ValueError(
            f"expected {num_scenes} image prompt(s) but parsed {len(prompts)}"
        )

    return prompts


def format_user_prompts(
    tokenizer,
    system_prompt: str,
    scenes: list[Scene],
    enable_thinking: bool = False,
) -> list[str]:
    scene_prompts = [
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Here is a scene JSON object. Produce a Stable Diffusion XL prompt\n"
                    f"{json.dumps(scene_payload(scene), ensure_ascii=False)}"
                ),
            },
        ]
        for scene in scenes
    ]

    return [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for messages in scene_prompts
    ]
