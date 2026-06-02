from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from utils import stage_1, vllm_wrapper
from utils.schema import Idea, PipelineConfig, load_config

_SRC_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = Path("configs/script_setup_qwen3_4b.yaml")

logger = logging.getLogger(__name__)


def write_ideas_jsonl(ideas: list[Idea], output_path: str) -> Path:
    """Write one JSON object per line using Idea.to_json()."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idea in ideas:
            handle.write(json.dumps(idea.to_json()) + "\n")
    logger.info("Wrote %s ideas to %s", len(ideas), path)
    return path


def run_stage_1(pipeline_config: PipelineConfig) -> list[Idea]:
    vcfg = pipeline_config.vllm_model_config
    idea_cfg = pipeline_config.idea_config

    model = vllm_wrapper.load_vllm_engine(
        dtype="auto",
        max_model_len=vcfg.max_model_len,
        model=vcfg.model_path,
        batch_size=vcfg.batch_size,
        gpu_memory_utilization=vcfg.gpu_memory_utilization,
        enforce_eager=vcfg.enforce_eager,
        tensor_parallel_size=vcfg.tensor_parallel_size,
        quantization=vcfg.quantization,
    )
    tokenizer = model.get_tokenizer()
    sampling_params = vllm_wrapper.sample_params(
        temperature=vcfg.temperature,
        max_tokens=vcfg.max_tokens,
    )

    ideas = stage_1.run_stage(
        logger,
        model,
        vcfg.model_path,
        sampling_params,
        tokenizer,
        idea_cfg,
    )
    write_ideas_jsonl(ideas, idea_cfg.output_path)
    return ideas


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    os.chdir(_SRC_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to script_setup YAML config",
    )
    args = parser.parse_args()

    pipeline_config = load_config(str(args.config))
    logger.info("Loaded config from %s", args.config)
    ideas = run_stage_1(pipeline_config)
    for idea in ideas:
        print(idea.to_json())


if __name__ == "__main__":
    main()
