"""Cross-stage validation for script_setup, image_setup, and vid_setup configs."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import yaml

_SRC_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_ROOT.parent
_PIPELINE_CONFIGS = _SRC_ROOT / "pipeline_configs"
_PROMPTED_VIDEO_TYPES = frozenset({"ltx", "sana", "cogvideox", "wan"})


def _clear_setup_utils_modules() -> None:
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]


def _load_setup_config(setup: str, config_rel: str) -> Any:
    """Load a setup YAML config without colliding ``utils`` package names."""
    setup_dir = _SRC_ROOT / setup
    config_path = setup_dir / config_rel
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _clear_setup_utils_modules()
    inserted = False
    setup_dir_str = str(setup_dir)
    if setup_dir_str not in sys.path:
        sys.path.insert(0, setup_dir_str)
        inserted = True
    try:
        schema = importlib.import_module("utils.schema")
        return schema.load_config(str(config_path))
    finally:
        if inserted:
            sys.path.remove(setup_dir_str)
        _clear_setup_utils_modules()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _normalize_data_path(value: str) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/") + "/"


def validate_bundle(
    script_config: str,
    image_config: str,
    vid_config: str,
    *,
    require_prompt_source: bool = True,
) -> list[str]:
    """Validate that three stage configs can run as one pipeline.

    Returns non-fatal warnings. Raises ``ValueError`` on hard incompatibilities.
    """
    script_cfg = _load_setup_config("script_setup", script_config)
    image_cfg = _load_setup_config("image_setup", image_config)
    vid_cfg = _load_setup_config("vid_setup", vid_config)

    warnings: list[str] = []
    errors: list[str] = []

    script_data_path = _normalize_data_path(script_cfg.image_config.script_path)
    image_data_path = _normalize_data_path(image_cfg.output_config.script_path)
    vid_data_path = _normalize_data_path(vid_cfg.io_config.script_path)

    if len({script_data_path, image_data_path, vid_data_path}) != 1:
        errors.append(
            "script_path mismatch across stages: "
            f"script_setup={script_data_path!r}, "
            f"image_setup={image_data_path!r}, "
            f"vid_setup={vid_data_path!r}"
        )

    image_subdir = image_cfg.output_config.output_subdir.strip("/")
    vid_subdir = vid_cfg.io_config.input_subdir.strip("/")
    if image_subdir != vid_subdir:
        errors.append(
            "image output_subdir must match vid input_subdir: "
            f"{image_subdir!r} != {vid_subdir!r}"
        )

    image_template = image_cfg.output_config.filename_template
    vid_image_template = vid_cfg.io_config.image_template
    if image_template != vid_image_template:
        errors.append(
            "scene image filename templates differ: "
            f"image_setup={image_template!r}, vid_setup={vid_image_template!r}"
        )

    video_type = vid_cfg.video_diffuser_config.normalized_type()
    gen_cfg = vid_cfg.generation_config
    if video_type == "svd":
        if gen_cfg.fps is None:
            errors.append("svd vid_setup generation_config requires fps for export")
    elif video_type in _PROMPTED_VIDEO_TYPES:
        if gen_cfg.frame_rate is None:
            errors.append(
                f"{video_type} vid_setup generation_config requires frame_rate for export"
            )
        fallback_prompt = (gen_cfg.prompt or "").strip()
        if require_prompt_source and not fallback_prompt:
            warnings.append(
                f"{video_type} video backend needs per-scene image_prompt in "
                "script.json (script_setup stage 4) or generation_config.prompt"
            )

    image_w = image_cfg.generation_config.width
    image_h = image_cfg.generation_config.height
    vid_w = gen_cfg.width
    vid_h = gen_cfg.height
    if image_w and image_h and (image_w != vid_w or image_h != vid_h):
        if video_type == "wan":
            warnings.append(
                f"image resolution {image_w}x{image_h} differs from vid target area "
                f"{vid_w}x{vid_h}; Wan preserves aspect ratio within that area"
            )
        else:
            warnings.append(
                f"image resolution {image_w}x{image_h} will be resized to "
                f"{vid_w}x{vid_h} for {video_type}"
            )

    if errors:
        raise ValueError(
            "Incompatible pipeline configs:\n"
            + "\n".join(f"  - {item}" for item in errors)
        )

    return warnings


def iter_setup_configs(setup: str) -> list[Path]:
    configs_dir = _SRC_ROOT / setup / "configs"
    return sorted(configs_dir.glob("*.yaml"))


def iter_pipeline_configs() -> list[Path]:
    return sorted(_PIPELINE_CONFIGS.glob("pipeline_*.yaml"))


def validate_all_setup_configs() -> None:
    """Load every stage YAML config to catch typos and backend constraint errors."""
    failures: list[str] = []
    for setup in ("script_setup", "image_setup", "vid_setup"):
        for path in iter_setup_configs(setup):
            rel = path.relative_to(_SRC_ROOT / setup).as_posix()
            try:
                _load_setup_config(setup, rel)
            except Exception as exc:
                failures.append(f"{setup}/{rel}: {exc}")
    if failures:
        raise ValueError(
            "Config load failures:\n" + "\n".join(f"  - {item}" for item in failures)
        )


def validate_all_pipeline_profiles() -> None:
    """Load and cross-validate every bundled pipeline profile."""
    failures: list[str] = []
    for path in iter_pipeline_configs():
        data = _load_yaml_mapping(path)
        try:
            validate_bundle(
                str(data["script_config"]).strip(),
                str(data["image_config"]).strip(),
                str(data["vid_config"]).strip(),
            )
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
    if failures:
        raise ValueError(
            "Pipeline profile validation failures:\n"
            + "\n".join(f"  - {item}" for item in failures)
        )
