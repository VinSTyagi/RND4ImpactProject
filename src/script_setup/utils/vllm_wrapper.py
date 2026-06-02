import logging

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def load_vllm_engine(
    dtype,
    max_model_len: int,
    model: str,
    batch_size: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    trust_remote_code: bool = True,
    quantization=None,
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
        "quantization": quantization,
        "trust_remote_code": trust_remote_code,
    }
    model = LLM(**args)
    logger.info("vLLM offline inference engine initialized")
    return model


def sample_params(temperature: float, max_tokens: int):
    sample = {"temperature": temperature, "max_tokens": max_tokens}
    params = SamplingParams(**sample)
    return params
