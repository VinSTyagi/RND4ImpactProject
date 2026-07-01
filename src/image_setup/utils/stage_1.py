from __future__ import annotations

import logging
from dataclasses import replace
from typing import Iterator

from utils import diffusion_wrapper
from utils.config import (
    ImageSetupPipelineConfig,
    refinement_active,
    scene_output_path,
    scene_raw_output_path,
    validate_pipeline_config,
)
from utils.schema import ImagePrompt, Scene, SceneScript

_PIPELINE_LOADERS = {
    "sdxl": diffusion_wrapper.txt2img_session,
    "sd15": diffusion_wrapper.txt2img_session,
}


def validate_scripts(scene_scripts: list[SceneScript]) -> None:
    """Ensure every scene script has image prompts before loading the pipeline."""
    errors: list[str] = []
    for scene_script in scene_scripts:
        scene = scene_script.scene
        prompts = scene_script.image_prompt
        if not prompts:
            errors.append(
                f"{scene_script.script_id}: scene {scene['scene_number']} missing image_prompt"
            )
    if errors:
        raise ValueError(
            "image generation requires image_prompt on every scene "
            "(run stage 5 of the script pipeline first):\n"
            + "\n".join(f"  - {item}" for item in errors)
            + "\n\nPopulate image_prompt in data/<script_id>/<scene>/script.json, "
            "then re-run image_setup."
        )


def iter_scenes(scene_script: SceneScript) -> Iterator[tuple[SceneScript, Scene]]:
    yield scene_script, scene_script.scene


def iter_scene_prompts(
    scene_script: SceneScript,
) -> Iterator[tuple[SceneScript, Scene, int, ImagePrompt]]:
    for scene_script_item, scene in iter_scenes(scene_script):
        prompts = scene_script_item.image_prompt or []
        for prompt_number, image_prompt in enumerate(prompts):
            yield scene_script_item, scene, prompt_number, image_prompt


def _latent_handoff_enabled(
    config: ImageSetupPipelineConfig,
    stages: list[int],
) -> bool:
    ref_cfg = config.refinement_config
    return (
        refinement_active(config)
        and 2 in stages
        and ref_cfg.type.strip().lower() == "sdxl_refiner"
    )


def run_stage(
    logger: logging.Logger,
    config: ImageSetupPipelineConfig,
    state: dict,
) -> dict:
    output_cfg = config.output_config
    gen_cfg = config.generation_config
    ref_cfg = config.refinement_config
    pipeline_type = config.pipeline_config.type.strip().lower()
    family = diffusion_wrapper.get_pipeline_family(pipeline_type)
    refine = refinement_active(config)
    stages: list[int] = state.get("stages", [1])
    use_latent_handoff = family.supports_latent_handoff and _latent_handoff_enabled(
        config, stages
    )

    validate_pipeline_config(config)

    session_factory = _PIPELINE_LOADERS.get(pipeline_type)
    if session_factory is None:
        supported = ", ".join(sorted(_PIPELINE_LOADERS))
        raise ValueError(
            f"unsupported pipeline type {pipeline_type!r}; supported: {supported}"
        )

    scene_scripts = SceneScript.read_all(output_cfg.script_path)
    logger.info(
        "Loaded %s scene script(s) from %s",
        len(scene_scripts),
        output_cfg.script_path,
    )
    validate_scripts(scene_scripts)

    written = 0
    skipped = 0
    latent_handoffs: dict[tuple[str, int, int], object] = {}

    with session_factory(
        config.pipeline_config,
        config.quantization_config,
        gen_cfg,
    ) as pipeline:
        pipeline_cfg = config.pipeline_config
        for scene_script in scene_scripts:
            for _, scene, prompt_number, image_prompt in iter_scene_prompts(scene_script):
                scene_number = scene["scene_number"]
                script_id = str(scene_script.script_id)

                if refine:
                    raw_path = scene_raw_output_path(
                        scene_script.script_id,
                        scene_number,
                        prompt_number,
                        output_cfg,
                    )
                    final_path = scene_output_path(
                        scene_script.script_id,
                        scene_number,
                        prompt_number,
                        output_cfg,
                    )
                    if output_cfg.skip_existing:
                        if final_path.is_file():
                            logger.info(
                                "Skipping script %s scene %s prompt %s (final exists): %s",
                                script_id,
                                scene_number,
                                prompt_number,
                                final_path,
                            )
                            skipped += 1
                            continue
                        if not use_latent_handoff and raw_path.is_file():
                            logger.info(
                                "Skipping script %s scene %s prompt %s (raw exists): %s",
                                script_id,
                                scene_number,
                                prompt_number,
                                raw_path,
                            )
                            skipped += 1
                            continue
                else:
                    raw_path = scene_output_path(
                        scene_script.script_id,
                        scene_number,
                        prompt_number,
                        output_cfg,
                    )
                    if output_cfg.skip_existing and raw_path.is_file():
                        logger.info(
                            "Skipping script %s scene %s prompt %s (output exists): %s",
                            script_id,
                            scene_number,
                            prompt_number,
                            raw_path,
                        )
                        skipped += 1
                        continue

                denoising_end = ref_cfg.denoising_end if use_latent_handoff else None
                output_type = "latent" if use_latent_handoff else "pil"
                seed_offset = scene_number * 10 + prompt_number
                effective_gen_cfg = gen_cfg
                if use_latent_handoff:
                    effective_gen_cfg = replace(
                        gen_cfg,
                        num_inference_steps=ref_cfg.num_inference_steps,
                    )

                result = diffusion_wrapper.generate_scene_image(
                    pipeline,
                    image_prompt,
                    effective_gen_cfg,
                    pipeline_type=pipeline_type,
                    pipeline_cfg=pipeline_cfg,
                    seed_offset=seed_offset,
                    output_type=output_type,
                    denoising_end=denoising_end,
                )

                if use_latent_handoff:
                    latent_handoffs[(script_id, scene_number, prompt_number)] = result
                    logger.info(
                        "Generated raw latent for script %s scene %s prompt %s",
                        script_id,
                        scene_number,
                        prompt_number,
                    )
                    if output_cfg.save_raw:
                        logger.info(
                            "Skipping raw_images PNG for script %s scene %s prompt %s "
                            "(latent handoff keeps tensors in memory for refinement)",
                            script_id,
                            scene_number,
                            prompt_number,
                        )
                else:
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    result.save(raw_path)
                    logger.info(
                        "Saved raw image for script %s scene %s prompt %s to %s",
                        script_id,
                        scene_number,
                        prompt_number,
                        raw_path,
                    )
                written += 1

    state["scene_scripts"] = scene_scripts
    state["latent_handoffs"] = latent_handoffs
    return {
        "scene_scripts": len(scene_scripts),
        "raw_written": written,
        "raw_skipped": skipped,
    }
