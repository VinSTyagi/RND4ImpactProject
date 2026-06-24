from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

_HUB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def resolve_hf_home() -> Path:
    if env_home := os.environ.get("HF_HOME"):
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface").resolve()


def is_hub_model_id(model_path: str) -> bool:
    return bool(_HUB_ID_RE.match(str(model_path).strip()))


def hub_repo_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.strip().replace("/", "--")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def model_paths_for_setup(setup_name: str, config_path: Path) -> list[str]:
    """Return Hugging Face repo ids declared in a setup YAML config."""
    data = _load_yaml(config_path)
    paths: list[str] = []

    if setup_name == "script_setup":
        section = data.get("global_vllm_config")
        if isinstance(section, dict) and section.get("model_path"):
            paths.append(str(section["model_path"]))
    elif setup_name == "image_setup":
        section = data.get("pipeline_config")
        if isinstance(section, dict):
            if section.get("model_path"):
                paths.append(str(section["model_path"]))
            if section.get("unet_checkpoint_repo"):
                paths.append(str(section["unet_checkpoint_repo"]))
    elif setup_name == "vid_setup":
        section = data.get("video_diffuser_config")
        if isinstance(section, dict) and section.get("model_path"):
            paths.append(str(section["model_path"]))
    else:
        raise ValueError(f"unknown setup name {setup_name!r}")

    return paths


def remove_hub_model(
    repo_id: str,
    *,
    hf_home: Path | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Delete one model snapshot directory from the local Hugging Face hub cache."""
    log = logger or logging.getLogger(__name__)
    if not is_hub_model_id(repo_id):
        return

    hf_home = hf_home or resolve_hf_home()
    hub = hf_home / "hub"
    cache_name = hub_repo_cache_name(repo_id)

    for path in (hub / cache_name, hub / ".locks" / cache_name):
        if not path.exists():
            continue
        shutil.rmtree(path, ignore_errors=True)
        log.info("Removed Hugging Face cache: %s", path)


def clear_setup_models(
    setup_name: str,
    config_path: Path | str,
    *,
    hf_home: Path | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Remove all hub models referenced by a setup config from the local cache."""
    log = logger or logging.getLogger(__name__)
    path = Path(config_path)
    if not path.is_file():
        log.warning("Config not found for cache cleanup: %s", path)
        return

    repo_ids = model_paths_for_setup(setup_name, path)
    if not repo_ids:
        return

    log.info("Clearing Hugging Face cache for %s model(s)", setup_name)
    seen: set[str] = set()
    for repo_id in repo_ids:
        if repo_id in seen:
            continue
        seen.add(repo_id)
        remove_hub_model(repo_id, hf_home=hf_home, logger=log)
