from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download

from utils.config import PipelineConfig
from utils.schema import resolve_path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logger = logging.getLogger(__name__)

_HUB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def collect_model_paths(config: PipelineConfig) -> list[str]:
    """Return the vLLM model path from the pipeline config."""
    return [config.global_vllm_config.model_path]


def _local_model_dir(model_path: str) -> Path | None:
    normalized = model_path.replace("\\", "/")
    if normalized.startswith("data/"):
        return resolve_path(model_path)
    path = Path(model_path)
    if path.is_absolute():
        return path
    if path.is_dir():
        return path
    return None


def _is_hub_model_id(model_path: str) -> bool:
    return bool(_HUB_ID_RE.match(model_path.strip()))


def prefetch_model(model_path: str) -> Path | None:
    """Download a Hugging Face model into the local cache, or skip local paths."""
    local_dir = _local_model_dir(model_path)
    if local_dir is not None:
        if local_dir.is_dir():
            logger.info("Using local model weights at %s", local_dir)
            return local_dir
        raise FileNotFoundError(f"local model directory not found: {local_dir}")

    if not _is_hub_model_id(model_path):
        raise ValueError(
            f"unsupported model_path {model_path!r}; expected a Hub id (org/name) "
            "or an existing local directory"
        )

    logger.info("Prefetching Hugging Face weights for %s", model_path)
    cache_dir = snapshot_download(repo_id=model_path)
    logger.info("Cached %s at %s", model_path, cache_dir)
    return Path(cache_dir)


def prefetch_models(model_paths: list[str]) -> list[Path]:
    """Prefetch each model path, skipping duplicates."""
    seen: set[str] = set()
    cached: list[Path] = []
    for model_path in model_paths:
        if model_path in seen:
            continue
        seen.add(model_path)
        result = prefetch_model(model_path)
        if result is not None:
            cached.append(result)
    return cached
