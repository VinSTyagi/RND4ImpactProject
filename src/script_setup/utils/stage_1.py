from __future__ import annotations

import logging

from prompts.prompt_reader import load_prompt_md
from utils.schema import Idea, IdeaConfig, parse_ideas_from_text
import datetime
import json
import re


def run_stage(
    logger: logging.Logger,
    model,
    sampling_params,
    tokenizer,
    config: IdeaConfig,
):
    logger.info("Beginning stage 1 (idea generation)")
    logger.info("Running vLLM generate for %s ideas", num_ideas)
    time = datetime.now()
    outputs = generate_ideas()

    logger.info(f"Generate ideas -- Time taken: {time - datetime.now()}")


def generate_ideas(logger, model, config, sampling_params, tokenizer):
    num_ideas = config.num_ideas
    sys_prompt = load_prompt_md(config.prompt_path)
    if not sys_prompt:
        raise FileNotFoundError(f"Prompt file not found or empty: {config.prompt_path}")

    user_prompt = f"Generate {num_ideas} story ideas."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    raw_outputs = model.generate([formatted_prompt], sampling_params)
    return clean_ideas(raw_outputs)


def _completion_text(raw_outputs) -> str:
    if isinstance(raw_outputs, str):
        return raw_outputs
    parts: list[str] = []
    for request_output in raw_outputs:
        for output in getattr(request_output, "outputs", ()):
            text = getattr(output, "text", None)
            if text:
                parts.append(text)
    if not parts:
        raise ValueError("no completion text found in model output")
    return parts[0] if len(parts) == 1 else "\n".join(parts)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _extract_json_array_text(text: str) -> str:
    cleaned = _strip_markdown_fences(text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in model output")
    return cleaned[start : end + 1]


def parse_ideas_from_text(text: str) -> list[Idea]:
    """Parse LLM text into validated Idea instances."""
    try:
        payload = json.loads(_extract_json_array_text(text))
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
            ideas.append(Idea.from_dict(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"idea at index {index}: {exc}") from exc
    return ideas


def clean_ideas(raw_outputs) -> list[Idea]:
    """Strip LLM wrappers and parse stage-1 output into Idea objects."""
    return parse_ideas_from_text(_completion_text(raw_outputs))
