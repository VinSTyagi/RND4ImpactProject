from __future__ import annotations

import json
import logging
import time
from uuid import UUID

from prompts.prompt_reader import load_prompt_md
from utils.llm_output import (
    extract_json_array_text,
    request_output_text,
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
    batch_size: int = 1,
    enable_thinking: bool = False,
) -> list[Script]:
    logger.info("Beginning stage 2 (title generation)")
    logger.info("Running vLLM generate for %s scripts", len(scripts))
    start = time.perf_counter()
    scripts = generate_titles(
        logger,
        model,
        scripts,
        config,
        sampling_params,
        tokenizer,
        batch_size=batch_size,
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
    batch_size: int = 1,
    enable_thinking: bool = False,
) -> list[Script]:
    if not scripts:
        raise ValueError("stage 2 received no scripts")
    prompt_path = resolve_path(config.prompt_path)
    sys_prompt = load_prompt_md(str(prompt_path))
    if not sys_prompt:
        raise FileNotFoundError(f"Prompt file not found or empty: {prompt_path}")

    model_name = model.llm_engine.model_config.model
    chunk_size = max(1, batch_size)
    max_retries = max(0, config.max_retries)
    scripts_by_id: dict[UUID, Script] = {s.script_id: s for s in scripts}
    pending = list(scripts)

    for attempt in range(max_retries + 1):
        if not pending:
            break
        if attempt > 0:
            logger.info(
                "Retrying title generation for %s script(s) (attempt %s/%s)",
                len(pending),
                attempt,
                max_retries,
            )

        chunks = [
            pending[i : i + chunk_size] for i in range(0, len(pending), chunk_size)
        ]
        prompts = [
            _format_prompt(
                config.num_words, tokenizer, sys_prompt, chunk, enable_thinking
            )
            for chunk in chunks
        ]
        logger.info("Submitting %s title prompt(s) to vLLM", len(prompts))
        generate_start = time.perf_counter()
        raw_outputs = model.generate(prompts, sampling_params)
        generate_elapsed = time.perf_counter() - generate_start
        logger.info("vLLM generate completed in %.2fs", generate_elapsed)

        next_pending: list[Script] = []
        for index, chunk in enumerate(chunks):
            if index >= len(raw_outputs):
                logger.warning(
                    "Missing vLLM output for batch of %s script(s)",
                    len(chunk),
                )
                next_pending.extend(chunk)
                continue

            request_output = raw_outputs[index]
            try:
                answer_text = strip_reasoning(request_output_text(request_output))
                titles = parse_titles_from_text(answer_text)
            except ValueError as exc:
                logger.warning(
                    "Failed to parse titles for batch of %s script(s): %s",
                    len(chunk),
                    exc,
                )
                next_pending.extend(chunk)
                continue

            matched = min(len(chunk), len(titles))
            if matched < len(chunk):
                logger.warning(
                    "Expected %s titles but parsed %s for this batch",
                    len(chunk),
                    len(titles),
                )
            for script, title in zip(chunk[:matched], titles[:matched]):
                script.raw_title = title
                script.model = model_name
                scripts_by_id[script.script_id] = script
            next_pending.extend(chunk[matched:])

        pending = next_pending

    if pending:
        missing_ids = ", ".join(str(script.script_id) for script in pending)
        raise ValueError(
            f"failed to generate titles for {len(pending)} script(s) after "
            f"{max_retries} retries: {missing_ids}"
        )

    return [scripts_by_id[script.script_id] for script in scripts]


def _format_prompt(
    num_words,
    tokenizer,
    sys_prompt: str,
    chunk: list[Script],
    enable_thinking: bool,
) -> str:
    payload = [idea_prompt_payload(script.idea) for script in chunk]
    user_prompt = (
        f"Here is a JSON array of {len(chunk)} story idea(s). Produce exactly one title of approximately {num_words} words per idea as a flat JSON array of strings in the same order.\n"
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


def parse_titles_from_text(text: str) -> list[str]:
    """Parse LLM text into a flat list of non-empty title strings."""
    try:
        payload = json.loads(extract_json_array_text(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in model output: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("expected a JSON array of title strings")

    titles: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"title at index {index} is not a non-empty string")
        titles.append(item.strip())
    if not titles:
        raise ValueError("JSON array contained no titles")
    return titles
