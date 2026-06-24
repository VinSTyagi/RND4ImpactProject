from __future__ import annotations

import argparse
import logging
import json
import os
import sys

# Must be set before vLLM is imported (see utils/vllm_wrapper.py).
os.environ.setdefault("VLLM_USE_V1", "0")

from pathlib import Path
from typing import Callable

from utils.schema import (
    PipelineConfig,
    Script,
    load_config,
)

_SCRIPT_SETUP_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _SCRIPT_SETUP_DIR.parent
_DEFAULT_CONFIG = Path("configs/script_setup_qwen3_4b.yaml")
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Ensure script_setup logs always reach stderr (vLLM may configure root first)."""
    log = logging.getLogger("script_setup.runner")
    log.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        log.addHandler(handler)
    log.propagate = False
    return log


logger = configure_logging()


def run_stage_1(pipeline_config: PipelineConfig, state: dict) -> dict:
    from utils import stage_1, vllm_wrapper

    vcfg = pipeline_config.global_vllm_config
    idea_cfg = pipeline_config.idea_config
    with vllm_wrapper.vllm_session(vcfg) as (model, sampling_params):
        tokenizer = model.get_tokenizer()
        scripts = stage_1.run_stage(
            logger,
            model,
            sampling_params,
            tokenizer,
            idea_cfg,
            enable_thinking=vcfg.enable_thinking,
        )
    for script in scripts:
        script.save(idea_cfg.output_path)
    logger.info("Saved %s scripts to %s", len(scripts), idea_cfg.output_path)
    state["scripts"] = scripts
    return state


def run_stage_2(pipeline_config: PipelineConfig, state: dict) -> dict:
    from utils import stage_2, vllm_wrapper

    vcfg = pipeline_config.global_vllm_config
    title_cfg = pipeline_config.title_config
    with vllm_wrapper.vllm_session(vcfg) as (model, sampling_params):
        tokenizer = model.get_tokenizer()
        scripts = stage_2.run_stage(
            logger,
            model,
            sampling_params,
            tokenizer,
            title_cfg,
            enable_thinking=vcfg.enable_thinking,
        )
    state["scripts"] = scripts
    return state


def run_stage_3(pipeline_config: PipelineConfig, state: dict) -> dict:
    from utils import stage_3, vllm_wrapper

    vcfg = pipeline_config.global_vllm_config
    scene_cfg = pipeline_config.scene_outline_config
    with vllm_wrapper.vllm_session(vcfg) as (model, sampling_params):
        tokenizer = model.get_tokenizer()
        scripts = stage_3.run_stage(
            logger,
            model,
            sampling_params,
            tokenizer,
            scene_cfg,
            enable_thinking=vcfg.enable_thinking,
        )
    state["scripts"] = scripts
    return state


def run_stage_4(pipeline_config: PipelineConfig, state: dict) -> dict:
    from utils import stage_4, vllm_wrapper

    vcfg = pipeline_config.global_vllm_config
    content_cfg = pipeline_config.scene_content_config
    with vllm_wrapper.vllm_session(vcfg) as (model, sampling_params):
        tokenizer = model.get_tokenizer()
        scripts = stage_4.run_stage(
            logger,
            model,
            sampling_params,
            tokenizer,
            content_cfg,
            enable_thinking=vcfg.enable_thinking,
        )
    state["scripts"] = scripts
    return state


def run_stage_5(pipeline_config: PipelineConfig, state: dict) -> dict:
    from utils import stage_5, vllm_wrapper

    vcfg = pipeline_config.global_vllm_config
    img_prompt_cfg = pipeline_config.image_config
    with vllm_wrapper.vllm_session(vcfg) as (model, sampling_params):
        tokenizer = model.get_tokenizer()
        scripts = stage_5.run_stage(
            logger,
            model,
            sampling_params,
            tokenizer,
            img_prompt_cfg,
            enable_thinking=vcfg.enable_thinking,
        )
    state["scripts"] = scripts
    return state


StageRunner = Callable[[PipelineConfig, dict], dict]

STAGES: dict[int, StageRunner] = {
    1: run_stage_1,
    2: run_stage_2,
    3: run_stage_3,
    4: run_stage_4,
    5: run_stage_5,
}


def resolve_stages(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> list[int]:
    """Map CLI flags to an ordered list of stage numbers to run."""
    if args.all:
        return sorted(STAGES)
    selected = [n for n in STAGES if getattr(args, f"stage_{n}")]
    if not selected:
        if args.prefetch:
            return []
        flags = ", ".join(f"--{n}" for n in sorted(STAGES))
        parser.error(f"Specify at least one stage: {flags}, or --all")
    return sorted(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one or more script_setup pipeline stages.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to script_setup YAML config",
    )
    for n in sorted(STAGES):
        parser.add_argument(
            f"--{n}",
            dest=f"stage_{n}",
            action="store_true",
            help=f"Run stage {n}",
        )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all implemented stages (overrides individual --N flags)",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Download Hugging Face weights for the configured vLLM model",
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
    os.chdir(_SCRIPT_SETUP_DIR)

    parser = build_parser()
    args = parser.parse_args()

    pipeline_config = load_config(str(args.config))
    vcfg = pipeline_config.global_vllm_config
    logger.info("Loaded config from %s", args.config)
    logger.info(
        "vLLM settings: model=%s quantization=%r max_model_len=%s",
        vcfg.model_path,
        vcfg.quantization,
        vcfg.max_model_len,
    )

    stages = resolve_stages(args, parser)

    if args.prefetch:
        from utils import prefetch

        model_paths = prefetch.collect_model_paths(pipeline_config)
        logger.info("Prefetching models: %s", ", ".join(model_paths))
        prefetch.prefetch_models(model_paths)
        if not stages:
            return

    if not stages:
        flags = ", ".join(f"--{n}" for n in sorted(STAGES))
        parser.error(f"Specify at least one stage: {flags}, or --all")
    logger.info("Running stages: %s", ", ".join(str(n) for n in stages))

    state: dict = {}
    ran_stages = False
    try:
        for n in stages:
            logger.info("=== Stage %s ===", n)
            STAGES[n](pipeline_config, state)
        ran_stages = True

        scripts = state.get("scripts")
        if scripts:
            for script in scripts:
                print(json.dumps(script.to_json()))
    finally:
        if ran_stages and os.environ.get("RND4IMPACT_KEEP_MODELS") != "1":
            if str(_SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(_SRC_ROOT))
            import hf_cache_cleanup

            hf_cache_cleanup.clear_setup_models(
                "script_setup",
                args.config,
                logger=logger,
            )


if __name__ == "__main__":
    main()
