from __future__ import annotations

import contextlib
import gc
import logging
from typing import TYPE_CHECKING

import torch
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

if TYPE_CHECKING:
    from utils.schema import VLLMModelConfig


logger = logging.getLogger(__name__)

_DISABLED = frozenset({"", "null", "none", "off", "false"})


def load_vllm_engine(
    dtype,
    max_model_len: int,
    model: str,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    max_num_batched_tokens: int,
    quantization: str,
    trust_remote_code: bool = True,
    tensor_parallel_size: int = 1,
):
    args = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "enforce_eager": enforce_eager,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    method = str(quantization).strip().lower()
    if method not in _DISABLED:
        args["quantization"] = method
    logger.info(
        "Loading vLLM model %s with quantization=%r (from YAML config)",
        model,
        args.get("quantization"),
    )
    llm = LLM(**args)
    logger.info(
        "vLLM offline inference engine initialized (max_num_seqs=%s)", max_num_seqs
    )
    return llm


def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_params(
    temperature: float,
    max_tokens: int,
    max_model_len: int,
    prompt_reserve: int = 512,
    repetition_penalty: float = 1.1,
    top_p: float = 0.9,
    top_k: int = -1,
    min_p: float = 0.0,
):
    cap = max(256, max_model_len - prompt_reserve)
    if max_tokens > cap:
        logger.warning(
            "max_tokens %s exceeds safe limit %s (max_model_len=%s); capping",
            max_tokens,
            cap,
            max_model_len,
        )
        max_tokens = cap
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    return params


@contextlib.contextmanager
def vllm_session(vcfg: VLLMModelConfig):
    """Load a vLLM engine for one stage and guarantee cleanup on exit."""
    model = load_vllm_engine(
        dtype="auto",
        max_model_len=vcfg.max_model_len,
        model=vcfg.model_path,
        max_num_seqs=vcfg.max_num_seqs,
        gpu_memory_utilization=vcfg.gpu_memory_utilization,
        enforce_eager=vcfg.enforce_eager,
        max_num_batched_tokens=vcfg.max_num_batched_tokens,
        tensor_parallel_size=vcfg.tensor_parallel_size,
        quantization=vcfg.quantization,
    )
    try:
        params = sample_params(
            temperature=vcfg.temperature,
            max_tokens=vcfg.max_tokens,
            max_model_len=vcfg.max_model_len,
            repetition_penalty=vcfg.repetition_penalty,
            top_p=vcfg.top_p,
            top_k=vcfg.top_k,
            min_p=vcfg.min_p,
        )
        yield model, params
    finally:
        del model
        cleanup()
