from __future__ import annotations

import json
import logging
import time

from prompts.prompt_reader import load_prompt_md
from utils.llm_helper import completion_text, extract_json_array_text, strip_reasoning
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
    logger.info("Running vLLM generate for %s ideas", config.num_ideas)
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

    user_prompt = (
        f"Generate exactly {num_ideas} distinct story ideas. "
        "Use thinking/reasoning if helpful, then output ONLY a complete JSON array "
        "that starts with `[` and ends with `]`."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    logger.info("Submitting prompt to vLLM (%s chars)", len(formatted_prompt))
    generate_start = time.perf_counter()
    raw_outputs = model.generate([formatted_prompt], sampling_params)
    generate_elapsed = time.perf_counter() - generate_start
    logger.info("vLLM generate completed in %.2fs", generate_elapsed)

    answer_text = strip_reasoning(completion_text(raw_outputs))
    logger.info("Model answer:\n%s", answer_text)

    ideas = parse_ideas_from_text(answer_text)
    model_name = model.llm_engine.model_config.model
    scripts = [Script(idea=idea, model=model_name) for idea in ideas]
    for script in scripts:
        script.idea["model"] = model_name
    if len(scripts) != num_ideas:
        logger.warning("Expected %s ideas but parsed %s", num_ideas, len(scripts))
    return scripts


def parse_ideas_from_text(text: str) -> list[Idea]:
    """Parse LLM text into validated idea dicts."""
    try:
        payload = json.loads(extract_json_array_text(text))
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
