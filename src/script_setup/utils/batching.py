from __future__ import annotations

import logging
from typing import Any


def generate_prompts(
    model: Any,
    prompts: list[str],
    sampling_params: Any,
    *,
    batch_size: int,
    logger: logging.Logger,
    label: str,
) -> list[Any]:
    """Run vLLM generate in chunks of ``batch_size`` prompts.

    Stage 3 passes one prompt per script; ``batch_size`` is scripts per vLLM call.
    Stages 4–5 pass one prompt per scene within each script; ``batch_size`` is
    scenes per vLLM call. ``batch_size <= 0`` sends all prompts in one call.
    """
    if not prompts:
        return []

    chunk_size = len(prompts) if batch_size <= 0 else batch_size
    outputs: list[Any] = []
    total = len(prompts)

    for start in range(0, total, chunk_size):
        chunk = prompts[start : start + chunk_size]
        end = start + len(chunk)
        logger.info(
            "%s: vLLM generate prompts %s-%s of %s (batch_size=%s)",
            label,
            start + 1,
            end,
            total,
            batch_size if batch_size > 0 else "all",
        )
        batch_outputs = model.generate(chunk, sampling_params)
        if len(batch_outputs) != len(chunk):
            raise ValueError(
                f"{label}: expected {len(chunk)} vLLM output(s) for prompts "
                f"{start + 1}-{end} but received {len(batch_outputs)}"
            )
        outputs.extend(batch_outputs)

    return outputs
