from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import hf_cache_cleanup
import pipeline_validate
import yaml

_REPO_ROOT = _SRC_ROOT.parent
_PIPELINE_CONFIGS = _SRC_ROOT / "pipeline_configs"
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_MIN_FREE_GB = 15.0

_PROFILE_FILES = {
    "default": _PIPELINE_CONFIGS / "pipeline_default.yaml",
    "moderate": _PIPELINE_CONFIGS / "pipeline_default.yaml",
    "t4": _PIPELINE_CONFIGS / "pipeline_t4.yaml",
    "40gb": _PIPELINE_CONFIGS / "pipeline_40gb.yaml",
    "a100": _PIPELINE_CONFIGS / "pipeline_a100.yaml",
}

_SETUP_PACKAGES = {
    "script_setup": "rnd4impact-script-setup",
    "image_setup": "rnd4impact-image-setup",
    "vid_setup": "rnd4impact-vid-setup",
}


@dataclass(frozen=True)
class SetupSpec:
    name: str
    package: str
    directory: Path
    runner: str
    config: str


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger("pipeline.runner")
    log.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        log.addHandler(handler)
    log.propagate = False
    return log


logger = configure_logging()


def load_pipeline_config(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline config must be a mapping: {path}")
    required = ("script_config", "image_config", "vid_config")
    missing = [key for key in required if not str(data.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Pipeline config {path} missing keys: {', '.join(missing)}")
    return {key: str(data[key]).strip() for key in required}


def resolve_configs(args: argparse.Namespace) -> tuple[str, str, str]:
    script_config = args.script_config
    image_config = args.image_config
    vid_config = args.vid_config

    if args.pipeline_config is not None:
        loaded = load_pipeline_config(args.pipeline_config)
        script_config = script_config or loaded["script_config"]
        image_config = image_config or loaded["image_config"]
        vid_config = vid_config or loaded["vid_config"]
    elif args.profile is not None:
        profile_path = _PROFILE_FILES.get(args.profile)
        if profile_path is None:
            choices = ", ".join(sorted(_PROFILE_FILES))
            raise ValueError(f"Unknown profile {args.profile!r}; choose one of: {choices}")
        loaded = load_pipeline_config(profile_path)
        script_config = script_config or loaded["script_config"]
        image_config = image_config or loaded["image_config"]
        vid_config = vid_config or loaded["vid_config"]

    if not script_config or not image_config or not vid_config:
        raise ValueError(
            "Specify --profile, --pipeline-config, or all of "
            "--script-config, --image-config, and --vid-config"
        )
    return script_config, image_config, vid_config


def build_setups(
    script_config: str,
    image_config: str,
    vid_config: str,
    *,
    run_script: bool,
    run_image: bool,
    run_vid: bool,
) -> list[SetupSpec]:
    setups: list[SetupSpec] = []
    if run_script:
        setups.append(
            SetupSpec(
                "script_setup",
                _SETUP_PACKAGES["script_setup"],
                _SRC_ROOT / "script_setup",
                "script_setup_runner.py",
                script_config,
            )
        )
    if run_image:
        setups.append(
            SetupSpec(
                "image_setup",
                _SETUP_PACKAGES["image_setup"],
                _SRC_ROOT / "image_setup",
                "image_setup_runner.py",
                image_config,
            )
        )
    if run_vid:
        setups.append(
            SetupSpec(
                "vid_setup",
                _SETUP_PACKAGES["vid_setup"],
                _SRC_ROOT / "vid_setup",
                "vid_setup_runner.py",
                vid_config,
            )
        )
    return setups


def venv_python() -> Path:
    if sys.platform == "win32":
        return _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return _REPO_ROOT / ".venv" / "bin" / "python"


def resolve_hf_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if env_home := os.environ.get("HF_HOME"):
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface").resolve()


def check_download_disk_space(hf_home: Path, *, min_free_gb: float = _MIN_FREE_GB) -> None:
    hf_home.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(hf_home)
    free_gb = usage.free / (1024**3)
    if free_gb >= min_free_gb:
        return
    raise RuntimeError(
        f"Only {free_gb:.1f} GB free for Hugging Face cache at {hf_home} "
        f"(need at least {min_free_gb:.0f} GB). "
        "Free space on $HOME or point downloads elsewhere, e.g. "
        "export HF_HOME=/scratch/$USER/hf_cache"
    )


def subprocess_env(hf_home: Path, *, keep_models: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(hf_home)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("VLLM_USE_V1", "0")
    if keep_models:
        env["RND4IMPACT_KEEP_MODELS"] = "1"
    else:
        env.pop("RND4IMPACT_KEEP_MODELS", None)
    return env


def sync_setup(
    spec: SetupSpec,
    *,
    dry_run: bool,
    env: dict[str, str],
) -> None:
    cmd = ["uv", "sync", "--package", spec.package]
    logger.info("Syncing %s dependencies", spec.name)
    logger.info("Command: %s (cwd=%s)", " ".join(cmd), _REPO_ROOT)
    if dry_run:
        return
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True, env=env)


def run_setup(
    spec: SetupSpec,
    extra_args: list[str],
    *,
    dry_run: bool,
    env: dict[str, str],
    keep_models: bool,
) -> None:
    python = venv_python()
    cmd = [str(python), spec.runner, "--config", spec.config, *extra_args]
    logger.info("=== %s ===", spec.name)
    logger.info("Command: %s (cwd=%s)", " ".join(cmd), spec.directory)
    if dry_run:
        return
    try:
        subprocess.run(cmd, cwd=spec.directory, check=True, env=env)
    finally:
        if not keep_models:
            hf_cache_cleanup.clear_setup_models(
                spec.name,
                spec.directory / spec.config,
                hf_home=Path(env["HF_HOME"]),
                logger=logger,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run script_setup, image_setup, and vid_setup sequentially "
            "(script → images → videos)."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=sorted(_PROFILE_FILES),
        help="Bundled config preset (default: 40gb)",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        help="YAML file with script_config, image_config, and vid_config paths",
    )
    parser.add_argument(
        "--script-config",
        help="script_setup YAML config (relative to src/script_setup/)",
    )
    parser.add_argument(
        "--image-config",
        help="image_setup YAML config (relative to src/image_setup/)",
    )
    parser.add_argument(
        "--vid-config",
        help="vid_setup YAML config (relative to src/vid_setup/)",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Prefetch script_setup vLLM weights before running stages",
    )
    parser.add_argument(
        "--skip-script",
        action="store_true",
        help="Skip script_setup (use existing data/<script_id>/script.json)",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip image_setup (use existing raw_images/)",
    )
    parser.add_argument(
        "--skip-vid",
        action="store_true",
        help="Skip vid_setup",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip cross-stage config compatibility checks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip uv sync before each setup (use current .venv state)",
    )
    parser.add_argument(
        "--hf-home",
        help="Hugging Face cache directory (default: $HF_HOME or ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--keep-models",
        action="store_true",
        help="Keep Hugging Face weights in cache after each setup finishes",
    )
    return parser


def main() -> None:
    global logger
    logger = configure_logging()
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        force=True,
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.profile is None and args.pipeline_config is None and not (
        args.script_config and args.image_config and args.vid_config
    ):
        args.profile = "40gb"

    try:
        script_config, image_config, vid_config = resolve_configs(args)
    except ValueError as exc:
        parser.error(str(exc))

    setups = build_setups(
        script_config,
        image_config,
        vid_config,
        run_script=not args.skip_script,
        run_image=not args.skip_image,
        run_vid=not args.skip_vid,
    )
    if not setups:
        parser.error("All setups were skipped; remove --skip-* flags")

    logger.info(
        "Pipeline configs: script=%s image=%s vid=%s",
        script_config,
        image_config,
        vid_config,
    )

    if not args.skip_validate:
        try:
            warnings = pipeline_validate.validate_bundle(
                script_config,
                image_config,
                vid_config,
                require_prompt_source=not args.skip_script,
            )
        except ValueError as exc:
            parser.error(str(exc))
        for warning in warnings:
            logger.warning("Config compatibility: %s", warning)

    hf_home = resolve_hf_home(args.hf_home)
    if not args.dry_run:
        check_download_disk_space(hf_home)
    env = subprocess_env(hf_home, keep_models=args.keep_models)
    logger.info(
        "Hugging Face cache: %s (HF_HUB_DISABLE_XET=%s)",
        hf_home,
        env.get("HF_HUB_DISABLE_XET", "0"),
    )

    for index, spec in enumerate(setups):
        if spec.name == "script_setup":
            extra_args = (
                ["--prefetch", "--all"]
                if args.prefetch and index == 0
                else ["--all"]
            )
        else:
            extra_args = []
        if not args.skip_sync:
            sync_setup(spec, dry_run=args.dry_run, env=env)
        run_setup(
            spec,
            extra_args,
            dry_run=args.dry_run,
            env=env,
            keep_models=args.keep_models,
        )

    if args.dry_run:
        logger.info("Dry run complete.")
    else:
        logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
