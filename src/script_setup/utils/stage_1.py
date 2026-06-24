from __future__ import annotations

import json
import logging
import time

from prompts.prompt_reader import load_prompt_md
from utils.batching import generate_prompts as run_llm_generations
from utils.llm_helper import completion_text, parse_json_array, strip_reasoning
from utils.schema import Idea, IdeaConfig, Script, resolve_path


def run_stage(
    logger: logging.Logger,
    model,
    sampling_params,
    tokenizer,
    config: IdeaConfig,
    enable_thinking: bool = True,
) -> list[Script]:
    logger.info("Beginning stage 1 (idea generation)")
    logger.info(
        "Running vLLM generate for %s ideas (batch_size=%s)",
        config.num_ideas,
        config.batch_size,
    )
    start = time.perf_counter()
    scripts = generate_scripts(
        logger,
        model,
        config,
        sampling_params,
        tokenizer,
        enable_thinking=enable_thinking,
    )
    elapsed = time.perf_counter() - start
    logger.info("Stage 1 total time: %.2fs", elapsed)
    return scripts


def _idea_counts(num_ideas: int, batch_size: int) -> list[int]:
    """Split ``num_ideas`` into per-prompt idea counts."""
    if batch_size <= 0:
        return [num_ideas]

    counts: list[int] = []
    remaining = num_ideas
    while remaining > 0:
        count = min(batch_size, remaining)
        counts.append(count)
        remaining -= count
    return counts


def _idea_user_prompt(count: int) -> str:
    noun = "idea" if count == 1 else "ideas"
    return (
        f"Generate exactly {count} distinct story {noun}. "
        "Each idea must be visually and narratively detailed in every field — "
        "rich setting, filmable hook, named protagonist with visible traits, "
        "and concrete stakes that later stages can expand into scenes and images. "
        "Use thinking/reasoning if helpful, then output ONLY a complete JSON array "
        "that starts with `[` and ends with `]`."
    )


def generate_scripts(
    logger: logging.Logger,
    model,
    config: IdeaConfig,
    sampling_params,
    tokenizer,
    enable_thinking: bool = True,
) -> list[Script]:
    num_ideas = config.num_ideas
    prompt_path = resolve_path(config.prompt_path)
    sys_prompt = load_prompt_md(str(prompt_path))
    if not sys_prompt:
        raise FileNotFoundError(f"Prompt file not found or empty: {prompt_path}")

    prompts = [
        _format_idea_prompt(
            tokenizer, sys_prompt, _idea_user_prompt(count), enable_thinking
        )
        for count in _idea_counts(num_ideas, config.batch_size)
    ]
    raw_outputs = run_llm_generations(
        model,
        prompts,
        sampling_params,
        batch_size=config.batch_size,
        logger=logger,
        label="stage 1",
    )

    ideas: list[Idea] = []
    for index, request_output in enumerate(raw_outputs, start=1):
        answer_text = strip_reasoning(completion_text([request_output]))
        logger.info("Model answer (batch %s/%s):\n%s", index, len(raw_outputs), answer_text)
        ideas.extend(parse_ideas_from_text(answer_text))

    model_name = model.llm_engine.model_config.model
    scripts = [Script(idea=idea, model=model_name) for idea in ideas]
    for script in scripts:
        script.idea["model"] = model_name
    if len(scripts) != num_ideas:
        logger.warning("Expected %s ideas but parsed %s", num_ideas, len(scripts))
    return scripts


def _format_idea_prompt(
    tokenizer,
    sys_prompt: str,
    user_prompt: str,
    enable_thinking: bool,
) -> str:
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


def parse_ideas_from_text(text: str) -> list[Idea]:
    """Parse LLM text into validated idea dicts."""
    try:
        payload = parse_json_array(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in model output: {exc}") from exc

    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("expected a JSON array of idea objects")

    if not items:
        raise ValueError("JSON array contained no ideas")

    ideas: list[Idea] = []
    for index, item in enumerate(items):
        try:
            ideas.append(Script.parse_idea_dict(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"idea at index {index}: {exc}") from exc
    return ideas


def clean_scripts(raw_outputs, model_name: str = "") -> list[Script]:
    """Strip LLM wrappers and parse stage-1 output into Script objects."""
    ideas = parse_ideas_from_text(strip_reasoning(completion_text(raw_outputs)))
    scripts = [Script(idea=idea, model=model_name) for idea in ideas]
    for script in scripts:
        if model_name:
            script.idea["model"] = model_name
    return scripts
