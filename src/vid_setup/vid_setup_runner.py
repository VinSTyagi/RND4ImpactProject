from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable

from utils import stage_1, stage_2
from utils.schema import VidSetupPipelineConfig, load_config, validate_pipeline_config

_VID_SETUP_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = Path("configs/vid_setup_svd.yaml")
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

StageRunner = Callable[[logging.Logger, VidSetupPipelineConfig, dict], dict]

STAGES: dict[int, StageRunner] = {
    1: stage_1.run_stage,
    2: stage_2.run_stage,
}


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Ensure vid_setup logs always reach stderr."""
    log = logging.getLogger("vid_setup.runner")
    log.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        log.addHandler(handler)
    log.propagate = False
    return log


logger = configure_logging()


def  resolve_stages(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> list[int]:
    """Map CLI flags to an ordered list of stage numbers to run."""
    if args.all:
        return sorted(STAGES)
    selected = [n for n in STAGES if getattr(args, f"stage_{n}")]
    if not selected:
        flags = ", ".join(f"--{n}" for n in sorted(STAGES))
        parser.error(f"Specify at least one stage: {flags}, or --all")
    return sorted(selected)


def build_parser() -> argparse.ArgumentParser:
    """_summary_
    Generates a argument parser used to ingest arguments and modify flow of code
    
    Returns:
        argparse.ArgumentParser: _description_
    """
    parser = argparse.ArgumentParser(
        description="Generate scene videos from refined images (SVD / LTX / AnimateDiff).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to vid_setup YAML config",
    )
    for n in sorted(STAGES):
        parser.add_argument(
            f"--{n}",
            dest=f"stage_{n}",
            action="store_true",
            help=(
                f"Run stage {n} "
                f"({'raw video generation' if n == 1 else 'video upscaling'})"
            ),
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
    os.chdir(_VID_SETUP_DIR)

    parser = build_parser()
    args = parser.parse_args()

    pipeline_config = load_config(str(args.config))
    validate_pipeline_config(pipeline_config)
    logger.info("Loaded config from %s", args.config)

    stages = resolve_stages(args, parser)
    if not stages:
        flags = ", ".join(f"--{n}" for n in sorted(STAGES))
        parser.error(f"Specify at least one stage: {flags}, or --all")
    logger.info("Running stages: %s", ", ".join(str(n) for n in stages))

    state: dict = {"stages": stages}
    summaries: dict[str, int] = {}

    for n in stages:
        logger.info("=== Stage %s ===", n)
        summaries.update(STAGES[n](logger, pipeline_config, state))

    logger.info(
        "Done: %s script(s), %s raw written, %s raw skipped, "
        "%s refined written, %s refined skipped",
        summaries.get("scripts", 0),
        summaries.get("raw_written", 0),
        summaries.get("raw_skipped", 0),
        summaries.get("refined_written", 0),
        summaries.get("refined_skipped", 0),
    )


if __name__ == "__main__":
    main()
