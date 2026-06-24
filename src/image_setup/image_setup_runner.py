from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from utils import stage_1
from utils.schema import ImageSetupPipelineConfig, load_config, validate_pipeline_config

_IMAGE_SETUP_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _IMAGE_SETUP_DIR.parent
_DEFAULT_CONFIG = Path("configs/image_setup_sdxl_fp16.yaml")
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate scene images from image_prompt fields in script.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to image_setup YAML config",
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

    state: dict = {}
    try:
        logger.info("=== image_setup ===")
        summaries = stage_1.run_stage(logger, pipeline_config, state)

        logger.info(
            "Done: %s script(s), %s written, %s skipped",
            summaries.get("scripts", 0),
            summaries.get("written", 0),
            summaries.get("skipped", 0),
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
