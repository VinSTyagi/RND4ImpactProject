from __future__ import annotations

import json
import logging
import time

from prompts.prompt_reader import load_prompt_md
from utils.llm_helper import (
    request_output_text,
    strip_markdown_fences,
    strip_reasoning,
)
from utils.schema import Script, TitleConfig, idea_prompt_payload, resolve_path


def run_stage(
    logger: logging.Logger,
    model,
    sampling_params,
    tokenizer,
    scripts: list[Script],
    config: TitleConfig,
    enable_thinking: bool = False,
) -> list[Script]:
    logger.info("Beginning stage 2 (title generation)")
    logger.info("Running vLLM generate for %s scripts", len(scripts))
    start = time.perf_counter()

    if not scripts:
        raise ValueError("stage 2 received no scripts")

    scripts = generate_titles(
        logger,
        model,
        scripts,
        config,
        sampling_params,
        tokenizer,
        enable_thinking=enable_thinking,
    )

    elapsed = time.perf_counter() - start
    logger.info("Stage 2 total time: %.2fs", elapsed)
    return scripts


def generate_titles(
    logger: logging.Logger,
    model,
    scripts: list[Script],
    config: TitleConfig,
    sampling_params,
    tokenizer,
    enable_thinking: bool = False,
) -> list[Script]:
    if not scripts:
        return scripts

    prompt_path = resolve_path(config.prompt_path)
    sys_prompt = load_prompt_md(str(prompt_path))
    if not sys_prompt:
        raise FileNotFoundError(f"Prompt file not found or empty: {prompt_path}")

    model_name = model.llm_engine.model_config.model
    prompts = [
        _format_prompt(config.num_words, tokenizer, sys_prompt, script, enable_thinking)
        for script in scripts
    ]
    logger.info("Submitting %s title prompt(s) to vLLM", len(prompts))
    generate_start = time.perf_counter()
    raw_outputs = model.generate(prompts, sampling_params)
    generate_elapsed = time.perf_counter() - generate_start
    logger.info("vLLM generate completed in %.2fs\n", generate_elapsed)

    if len(raw_outputs) != len(scripts):
        raise ValueError(
            f"expected {len(scripts)} vLLM output(s) but received {len(raw_outputs)}"
        )

    for script, request_output in zip(scripts, raw_outputs):
        try:
            answer_text = strip_reasoning(request_output_text(request_output))
            logger.info(
                "Parsed title output for %s:\n%s", script.script_id, answer_text
            )
            script.raw_title = parse_title_from_text(answer_text)
        except ValueError as exc:
            raise ValueError(
                f"failed to parse title for script {script.script_id}: {exc}"
            ) from exc
        script.model = model_name

    return scripts


def _format_prompt(
    num_words,
    tokenizer,
    sys_prompt: str,
    script: Script,
    enable_thinking: bool,
) -> str:
    payload = idea_prompt_payload(script.idea)
    user_prompt = (
        f"Here is a story idea JSON object. Produce exactly one title string of approximately {num_words} words.\n"
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


def parse_title_from_text(text: str) -> str:
    """Parse LLM text into a single non-empty title string."""
    cleaned = strip_markdown_fences(text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in model output: {exc}") from exc

    if isinstance(payload, str):
        title = payload.strip()
    elif isinstance(payload, dict):
        raw_title = payload.get("title")
        if not isinstance(raw_title, str):
            raise ValueError(
                "expected a JSON string title or object with a string 'title' field"
            )
        title = raw_title.strip()
    else:
        raise ValueError("expected a JSON string title")

    if not title:
        raise ValueError("title must be non-empty")
    return title
