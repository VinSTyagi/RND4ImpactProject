from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
_SETUPS = ("script_setup", "image_setup", "vid_setup", "audio_setup")


def _clear_setup_utils_modules() -> None:
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]


def _load_setup_config(setup: str, config_rel: str) -> object:
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
        config = importlib.import_module("utils.config")
        return config.load_config(str(config_path))
    finally:
        if inserted:
            sys.path.remove(setup_dir_str)
        _clear_setup_utils_modules()


def _iter_setup_configs(setup: str) -> list[Path]:
    return sorted((_SRC_ROOT / setup / "configs").glob("*.yaml"))


def test_all_stage_configs_load() -> None:
    """Load every stage YAML config to catch typos and backend constraint errors."""
    failures: list[str] = []
    for setup in _SETUPS:
        for path in _iter_setup_configs(setup):
            rel = path.relative_to(_SRC_ROOT / setup).as_posix()
            try:
                _load_setup_config(setup, rel)
            except Exception as exc:
                failures.append(f"{setup}/{rel}: {exc}")
    if failures:
        raise AssertionError(
            "Config load failures:\n" + "\n".join(f"  - {item}" for item in failures)
        )
