from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable

from utils import stage_1, stage_2
from utils.config import (
    ImageSetupPipelineConfig,
    load_config,
    refinement_active,
    validate_pipeline_config,
)

_IMAGE_SETUP_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _IMAGE_SETUP_DIR.parent
_DEFAULT_CONFIG = Path("configs/image_setup_12gb.yaml")
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

StageRunner = Callable[[logging.Logger, ImageSetupPipelineConfig, dict], dict]

STAGES: dict[int, StageRunner] = {
    1: stage_1.run_stage,
    2: stage_2.run_stage,
}


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Ensure image_setup logs always reach stderr."""
    log = logging.getLogger("image_setup.runner")
    log.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        log.addHandler(handler)
    log.propagate = False
    return log


logger = configure_logging()


def resolve_stages(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    pipeline_config: ImageSetupPipelineConfig,
) -> list[int]:
    """Map CLI flags to an ordered list of stage numbers to run."""
    if args.all:
        return sorted(STAGES)
    selected = [n for n in STAGES if getattr(args, f"stage_{n}")]
    if not selected:
        if refinement_active(pipeline_config):
            return sorted(STAGES)
        return [1]
    return sorted(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate scene images from image_prompt fields in per-scene script.json "
            "(produced by script_setup stage 5)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to image_setup YAML config",
    )
    for n in sorted(STAGES):
        parser.add_argument(
            f"--{n}",
            dest=f"stage_{n}",
            action="store_true",
            help=f"Run stage {n} (1=raw generation, 2=refinement)",
        )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all implemented stages (overrides individual --N flags)",
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
    os.chdir(_IMAGE_SETUP_DIR)

    parser = build_parser()
    args = parser.parse_args()

    pipeline_config = load_config(str(args.config))
    validate_pipeline_config(pipeline_config)
    logger.info("Loaded config from %s", args.config)

    stages = resolve_stages(args, parser, pipeline_config)
    logger.info("Running stages: %s", ", ".join(str(n) for n in stages))

    state: dict = {"stages": stages}
    summaries: dict[str, int] = {}

    try:
        for n in stages:
            logger.info("=== Stage %s ===", n)
            result = STAGES[n](logger, pipeline_config, state)
            summaries.update(result)

        logger.info(
            "Done: %s script(s), %s raw written, %s raw skipped, "
            "%s refined written, %s refined skipped",
            summaries.get("scripts", 0),
            summaries.get("raw_written", 0),
            summaries.get("raw_skipped", 0),
            summaries.get("refined_written", 0),
            summaries.get("refined_skipped", 0),
        )
    finally:
        if os.environ.get("RND4IMPACT_KEEP_MODELS") != "1":
            if str(_SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(_SRC_ROOT))
            import hf_cache_cleanup

            hf_cache_cleanup.clear_setup_models(
                "image_setup",
                args.config,
                logger=logger,
            )


if __name__ == "__main__":
    main()
