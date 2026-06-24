"""Normalize SD image prompts for CLIP token limits (image_setup inference)."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

CLIP_TARGET_TOKENS = 60
CLIP_HARD_LIMIT = 77
REASONING_MAX_WORDS = 25
NEGATIVE_SCENE_EXCLUSION_LIMIT = 5

VALID_STYLE_PRESETS = frozenset(
    {
        "cinematic",
        "fantasy-art",
        "comic-book",
        "analog-film",
        "neon-punk",
        "dark-gothic",
        "painterly",
        "photorealistic",
    }
)

VALID_ASPECT_RATIOS = frozenset({"16:9", "9:16", "1:1"})

NEGATIVE_PROMPT_BASE = (
    "blurry",
    "low quality",
    "bad anatomy",
    "deformed",
    "extra hands",
    "extra arms",
    "extra legs",
    "watermark",
    "text",
    "cartoon",
    "anime",
)

_POSITIVE_QUALITY_SPAM = frozenset(
    {
        "masterpiece",
        "best quality",
        "high quality",
        "highly detailed",
        "ultra detailed",
        "sharp focus",
        "8k",
        "8k resolution",
        "4k",
        "4k resolution",
        "hdr",
        "absurdres",
        "incredibly absurdres",
        "professional",
        "award winning",
    }
)


@lru_cache(maxsize=1)
def _clip_tokenizer():
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


def clip_token_count(text: str) -> int:
    if not text.strip():
        return 0
    tokenizer = _clip_tokenizer()
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def join_prompt_tags(tags: list[str]) -> str:
    return ", ".join(tag.strip() for tag in tags if tag.strip())


def coerce_prompt_tags(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        tags = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        tags = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise ValueError(f"{field} must be a string or non-empty array")
    if not tags:
        raise ValueError(f"{field} must not be empty")
    return tags


def _truncate_text_to_clip(text: str, max_tokens: int) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return cleaned
    if clip_token_count(cleaned) <= max_tokens:
        return cleaned

    words = cleaned.split()
    while len(words) > 1:
        words.pop()
        candidate = " ".join(words)
        if clip_token_count(candidate) <= max_tokens:
            return candidate

    while cleaned and clip_token_count(cleaned) > max_tokens:
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def truncate_tags_to_clip(
    tags: list[str], max_tokens: int = CLIP_TARGET_TOKENS
) -> list[str]:
    if not tags:
        return tags

    original = join_prompt_tags(tags)
    working = [tag.strip() for tag in tags if tag.strip()]
    while working:
        if clip_token_count(join_prompt_tags(working)) <= max_tokens:
            result = working
            break
        working.pop()
    else:
        fallback = _truncate_text_to_clip(tags[0], max_tokens)
        result = [fallback] if fallback else tags[:1]

    trimmed = join_prompt_tags(result)
    if trimmed != original:
        logger.info(
            "Trimmed prompt from %s to %s CLIP tokens",
            clip_token_count(original),
            clip_token_count(trimmed),
        )
    return result


def normalize_positive_tags(tags: list[str]) -> list[str]:
    filtered: list[str] = []
    for tag in tags:
        key = tag.strip().lower()
        if not key or key in _POSITIVE_QUALITY_SPAM:
            continue
        filtered.append(tag.strip())
    if not filtered:
        filtered = [tags[0].strip()]
    return truncate_tags_to_clip(filtered)


def normalize_negative_tags(tags: list[str]) -> list[str]:
    base_keys = {tag.lower() for tag in NEGATIVE_PROMPT_BASE}
    merged: list[str] = list(NEGATIVE_PROMPT_BASE)
    seen = set(base_keys)
    scene_added = 0

    for tag in tags:
        cleaned = tag.strip()
        key = cleaned.lower()
        if not key or key in seen:
            continue
        if scene_added >= NEGATIVE_SCENE_EXCLUSION_LIMIT:
            break
        merged.append(cleaned)
        seen.add(key)
        scene_added += 1

    return truncate_tags_to_clip(merged)


def normalize_style_preset(value: Any) -> str:
    style = str(value).strip().lower()
    if style not in VALID_STYLE_PRESETS:
        valid = ", ".join(sorted(VALID_STYLE_PRESETS))
        raise ValueError(f"invalid style_preset {value!r}; expected one of: {valid}")
    return style


def normalize_aspect_ratio(value: Any) -> str:
    ratio = str(value).strip()
    if ratio not in VALID_ASPECT_RATIOS:
        valid = ", ".join(sorted(VALID_ASPECT_RATIOS))
        raise ValueError(f"invalid aspect_ratio {value!r}; expected one of: {valid}")
    return ratio


def normalize_cfg_scale(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("cfg_scale must be a number")
    if isinstance(value, (int, float)):
        number = int(round(float(value)))
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError("cfg_scale must be a non-empty string")
        try:
            number = int(round(float(raw)))
        except ValueError as exc:
            raise ValueError(f"invalid cfg_scale: {value!r}") from exc
    number = max(5, min(12, number))
    return str(number)


def normalize_reasoning(value: Any) -> str:
    reasoning = re.sub(r"\s+", " ", str(value).strip())
    if not reasoning:
        raise ValueError("reasoning must be a non-empty string")
    words = reasoning.split()
    if len(words) > REASONING_MAX_WORDS:
        reasoning = " ".join(words[:REASONING_MAX_WORDS]).rstrip(".,;:")
        if reasoning and reasoning[-1] not in ".!?":
            reasoning += "."
    return reasoning


def normalize_image_prompt_fields(
    *,
    positive_prompt: list[str],
    negative_prompt: list[str],
    style_preset: Any,
    aspect_ratio: Any,
    cfg_scale: Any,
    reasoning: Any,
) -> dict[str, Any]:
    """Apply prompt constraints when reading ImagePrompt fields from script.json."""
    return {
        "positive_prompt": normalize_positive_tags(positive_prompt),
        "negative_prompt": normalize_negative_tags(negative_prompt),
        "style_preset": normalize_style_preset(style_preset),
        "aspect_ratio": normalize_aspect_ratio(aspect_ratio),
        "cfg_scale": normalize_cfg_scale(cfg_scale),
        "reasoning": normalize_reasoning(reasoning),
    }
