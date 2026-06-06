from __future__ import annotations

import re

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


def extract_json_array_text(text: str) -> str:
    cleaned = strip_markdown_fences(text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in model output")
    return cleaned[start : end + 1]
