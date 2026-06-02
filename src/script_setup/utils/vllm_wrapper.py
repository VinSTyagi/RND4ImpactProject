from __future__ import annotations

import logging

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

_DEFAULT_QUANTIZATION = "awq"
_DISABLED = frozenset({"", "null", "none", "off", "false"})


def load_vllm_engine(
    dtype,
    max_model_len: int,
    model: str,
    batch_size: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    trust_remote_code: bool = True,
    quantization: str = _DEFAULT_QUANTIZATION,
    tensor_parallel_size: int = 1,
):
    args = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "enforce_eager": enforce_eager,
        "max_num_seqs": batch_size,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    method = str(quantization).strip().lower()
    if method not in _DISABLED:
        args["quantization"] = method
    llm = LLM(**args)
    logger.info("vLLM offline inference engine initialized")
    return llm


def sample_params(temperature: float, max_tokens: int):
    sample = {"temperature": temperature, "max_tokens": max_tokens}
    params = SamplingParams(**sample)
    return params
