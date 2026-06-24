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
    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")
    json_starts = [index for index in (obj_start, arr_start) if index != -1]
    if json_starts:
        start = min(json_starts)
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


def _escape_control_chars_in_json_strings(text: str) -> str:
    """Escape raw newlines/tabs inside JSON string literals."""
    result: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
                result.append(char)
            elif char == "\\":
                escape = True
                result.append(char)
            elif char == '"':
                in_string = False
                result.append(char)
            elif char == "\n":
                result.append("\\n")
            elif char == "\r":
                result.append("\\r")
            elif char == "\t":
                result.append("\\t")
            else:
                result.append(char)
        else:
            if char == '"':
                in_string = True
            result.append(char)
    return "".join(result)


def _insert_missing_commas_between_objects(text: str) -> str:
    return re.sub(r"\}(\s*)\{", r"},\1{", text)


def _repair_interior_double_quotes(text: str) -> str:
    """Escape double quotes that appear inside JSON string values."""
    result: list[str] = []
    in_string = False
    escape = False

    for char in text:
        if not in_string:
            if char == '"':
                in_string = True
            result.append(char)
            continue

        if escape:
            escape = False
            result.append(char)
            continue
        if char == "\\":
            escape = True
            result.append(char)
            continue
        if char == '"':
            index = len(result) + 1
            while index < len(text) and text[index] in " \t\n\r":
                index += 1
            if index < len(text) and text[index] in ",:]}":
                in_string = False
                result.append(char)
            else:
                result.append('\\"')
            continue

        result.append(char)

    return "".join(result)


def _json_repair_candidates(text: str) -> list[str]:
    normalized = _normalize_json_quotes(text)
    seen: set[str] = set()
    candidates: list[str] = []

    def add(value: str) -> None:
        if value not in seen:
            seen.add(value)
            candidates.append(value)

    escaped = _escape_control_chars_in_json_strings(normalized)
    repaired_quotes = _repair_interior_double_quotes(normalized)
    repaired_quotes_escaped = _escape_control_chars_in_json_strings(repaired_quotes)
    for base in (normalized, escaped, repaired_quotes, repaired_quotes_escaped):
        add(base)
        add(_remove_trailing_commas(base))
        with_commas = _insert_missing_commas_between_objects(base)
        add(with_commas)
        add(_remove_trailing_commas(with_commas))
    return candidates


def _try_loads_json_text(text: str) -> Any:
    last_exc: json.JSONDecodeError | None = None
    for candidate in _json_repair_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
    assert last_exc is not None
    raise last_exc


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


def extract_json_object_texts(text: str) -> list[str]:
    """Extract complete top-level object literals from a JSON array fragment."""
    cleaned = strip_markdown_fences(text)
    start = cleaned.find("[")
    index = start + 1 if start != -1 else 0

    objects: list[str] = []
    length = len(cleaned)
    while index < length:
        while index < length and cleaned[index] in " \t\n\r,":
            index += 1
        if index >= length or cleaned[index] == "]":
            break
        if cleaned[index] != "{":
            index += 1
            continue

        obj_start = index
        depth = 0
        in_string = False
        escape = False
        parsed = False
        for pos in range(index, length):
            char = cleaned[pos]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        objects.append(cleaned[obj_start : pos + 1])
                        index = pos + 1
                        parsed = True
                        break
        if not parsed:
            break
    return objects


def _parse_json_objects_fallback(text: str) -> list[Any] | None:
    objects = extract_json_object_texts(text)
    if not objects:
        return None
    return [_try_loads_json_text(obj_text) for obj_text in objects]


def loads_json_text(text: str) -> Any:
    """Parse JSON text with light repairs for common LLM formatting mistakes."""
    return _try_loads_json_text(text)


def extract_json_object_text(text: str) -> str:
    cleaned = strip_markdown_fences(text)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")

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
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]

    raise ValueError("no complete JSON object found in model output")


def parse_json_object(text: str) -> Any:
    """Extract and parse the first top-level JSON object from LLM output."""
    cleaned = strip_markdown_fences(text)
    try:
        object_text = extract_json_object_text(text)
    except ValueError:
        return _try_loads_json_text(cleaned)
    return _try_loads_json_text(object_text)


def parse_json_array(text: str) -> Any:
    """Extract and parse the first top-level JSON array from LLM output."""
    cleaned = strip_markdown_fences(text)
    last_exc: Exception | None = None

    try:
        array_text = extract_json_array_text(text)
    except ValueError as exc:
        last_exc = exc
    else:
        try:
            return _try_loads_json_text(array_text)
        except json.JSONDecodeError as exc:
            last_exc = exc

    fallback = _parse_json_objects_fallback(cleaned)
    if fallback:
        return fallback

    if last_exc is not None:
        raise last_exc
    raise ValueError("no JSON array found in model output")
