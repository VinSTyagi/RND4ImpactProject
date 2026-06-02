from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from utils import stage_1, vllm_wrapper
from utils.schema import PipelineConfig, load_config

_SRC_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = Path("configs/script_setup.yaml")

logger = logging.getLogger(__name__)


def run_stage_1(pipeline_config: PipelineConfig):
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
        quantization=vcfg.quantization or None,
    )
    tokenizer = vllm_wrapper.load_tokenizer(model)
    sampling_params = vllm_wrapper.sample_params(
        temperature=vcfg.temperature,
        max_tokens=vcfg.max_tokens,
    )

    outputs = stage_1.run_stage(logger, model, sampling_params, tokenizer, idea_cfg)

    return outputs


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
    print(ideas)


if __name__ == "__main__":
    main()
