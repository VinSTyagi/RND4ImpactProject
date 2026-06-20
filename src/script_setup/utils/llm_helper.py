from __future__ import annotations

import json
import re
from typing import Any

_THINKING_BLOCK_RES = (
    re.compile(
        r"<\s*(?:redacted_)?think(?:ing)?\s*>[\s\S]*?<\s*/\s*(?:redacted_)?think(?:ing)?\s*>",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\s*(?:redacted_)?think(?:ing)?\s*>[\s\S]*",
        re.IGNORECASE,
    ),
)


def completion_text(raw_outputs) -> str:
    """Concatenate text across the RequestOutputs of a single-prompt generate call."""
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


def request_output_text(request_output) -> str:
    """Return the first completion text for a single batched RequestOutput."""
    for output in getattr(request_output, "outputs", ()):
        text = getattr(output, "text", None)
        if text:
            return text
    raise ValueError("no completion text found in request output")


def strip_reasoning(text: str) -> str:
    """Drop model chain-of-thought; keep only the post-reasoning answer for parsing."""
    cleaned = text.strip()
    for pattern in _THINKING_BLOCK_RES:
        cleaned = pattern.sub("", cleaned).strip()
    closing = re.search(
        r"<\s*/\s*(?:redacted_)?think(?:ing)?\s*>",
        cleaned,
        re.IGNORECASE,
    )
    if closing:
        cleaned = cleaned[closing.end() :].strip()
    start = cleaned.find("[")
    if start > 0:
        cleaned = cleaned[start:]
    return cleaned


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _normalize_json_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _remove_trailing_commas(text: str) -> str:
    prev = None
    repaired = text
    while repaired != prev:
        prev = repaired
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def extract_json_array_text(text: str) -> str:
    cleaned = strip_markdown_fences(text)
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("no JSON array found in model output")

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(cleaned[start:], start):
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

    raise ValueError("no complete JSON array found in model output")


def loads_json_text(text: str) -> Any:
    """Parse JSON text with light repairs for common LLM formatting mistakes."""
    normalized = _normalize_json_quotes(text)
    candidates = [normalized, _remove_trailing_commas(normalized)]
    last_exc: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def parse_json_array(text: str) -> Any:
    """Extract and parse the first top-level JSON array from LLM output."""
    return loads_json_text(extract_json_array_text(text))
