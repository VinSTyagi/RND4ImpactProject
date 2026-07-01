from __future__ import annotations

import logging

from PIL import Image

from utils import diffusion_wrapper
from utils.config import (
    ImageSetupPipelineConfig,
    refinement_active,
    scene_output_path,
    scene_raw_output_path,
    validate_pipeline_config,
)
from utils.schema import SceneScript
from utils.stage_1 import iter_scene_prompts, validate_scripts


def _validate_raw_inputs(
    config: ImageSetupPipelineConfig,
    scene_scripts: list[SceneScript],
    latent_handoffs: dict[tuple[str, int, int], object],
) -> None:
    output_cfg = config.output_config
    missing: list[str] = []
    for scene_script in scene_scripts:
        for _, scene, prompt_number, _ in iter_scene_prompts(scene_script):
            key = (str(scene_script.script_id), scene["scene_number"], prompt_number)
            if key in latent_handoffs:
                continue
            raw_path = scene_raw_output_path(
                scene_script.script_id,
                scene["scene_number"],
                prompt_number,
                output_cfg,
            )
            if not raw_path.is_file():
                missing.append(str(raw_path))
    if missing:
        raise ValueError(
            "refinement requires stage 1 raw images; missing:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\n\nRun stage 1 first:\n"
            "  python image_setup_runner.py --config <config> --1"
        )


def run_stage(
    logger: logging.Logger,
    config: ImageSetupPipelineConfig,
    state: dict,
) -> dict:
    ref_cfg = config.refinement_config
    output_cfg = config.output_config
    gen_cfg = config.generation_config
    pipeline_type = config.pipeline_config.type.strip().lower()
    family = diffusion_wrapper.get_pipeline_family(pipeline_type)

    validate_pipeline_config(config)

    if not refinement_active(config):
        logger.info("Refinement disabled (type=%s); skipping stage 2", ref_cfg.type)
        return {
            "scene_scripts": 0,
            "refined_written": 0,
            "refined_skipped": 0,
        }

    ref_type = ref_cfg.type.strip().lower()
    scene_scripts = SceneScript.read_all(output_cfg.script_path)
    logger.info(
        "Loaded %s scene script(s) from %s",
        len(scene_scripts),
        output_cfg.script_path,
    )
    validate_scripts(scene_scripts)

    latent_handoffs: dict[tuple[str, int, int], object] = state.get("latent_handoffs", {})
    _validate_raw_inputs(config, scene_scripts, latent_handoffs)

    if ref_type == "sdxl_refiner":
        if not family.supports_refiner:
            raise ValueError(
                f"{pipeline_type} pipelines do not support sdxl_refiner refinement"
            )
        model_path = ref_cfg.model_path
        scheduler = ref_cfg.scheduler
    elif ref_type == "img2img":
        model_path = config.pipeline_config.model_path
        scheduler = config.pipeline_config.scheduler
    else:
        raise ValueError(
            f"unsupported refinement type {ref_cfg.type!r}; "
            "expected sdxl_refiner, img2img, or none"
        )

    written = 0
    skipped = 0

    pipeline_cfg = config.pipeline_config
    with diffusion_wrapper.img2img_session(
        model_path,
        pipeline_cfg,
        config.quantization_config,
        gen_cfg,
        scheduler=scheduler,
        variant=ref_cfg.variant if ref_type == "sdxl_refiner" else None,
    ) as pipeline:
        for scene_script in scene_scripts:
            for _, scene, prompt_number, image_prompt in iter_scene_prompts(scene_script):
                scene_number = scene["scene_number"]
                script_id = str(scene_script.script_id)
                final_path = scene_output_path(
                    scene_script.script_id,
                    scene_number,
                    prompt_number,
                    output_cfg,
                )

                if output_cfg.skip_existing and final_path.is_file():
                    logger.info(
                        "Skipping refinement for script %s scene %s prompt %s (final exists): %s",
                        script_id,
                        scene_number,
                        prompt_number,
                        final_path,
                    )
                    skipped += 1
                    continue

                key = (script_id, scene_number, prompt_number)
                if key in latent_handoffs:
                    image_input = latent_handoffs[key]
                else:
                    raw_path = scene_raw_output_path(
                        scene_script.script_id,
                        scene_number,
                        prompt_number,
                        output_cfg,
                    )
                    image_input = Image.open(raw_path).convert("RGB")

                seed_offset = scene_number * 10 + prompt_number
                if ref_type == "sdxl_refiner":
                    refined = diffusion_wrapper.refine_scene_sdxl_refiner(
                        pipeline,
                        image_prompt,
                        gen_cfg,
                        ref_cfg,
                        pipeline_cfg,
                        image_input,
                        seed_offset=seed_offset,
                    )
                else:
                    if not isinstance(image_input, Image.Image):
                        raise TypeError("img2img refinement requires a PIL image input")
                    refined = diffusion_wrapper.refine_scene_img2img(
                        pipeline,
                        image_prompt,
                        gen_cfg,
                        ref_cfg,
                        pipeline_type=pipeline_type,
                        pipeline_cfg=pipeline_cfg,
                        image=image_input,
                        seed_offset=seed_offset,
                    )

                final_path.parent.mkdir(parents=True, exist_ok=True)
                refined.save(final_path)
                logger.info(
                    "Saved refined image for script %s scene %s prompt %s to %s",
                    script_id,
                    scene_number,
                    prompt_number,
                    final_path,
                )
                written += 1

    return {
        "scene_scripts": len(scene_scripts),
        "refined_written": written,
        "refined_skipped": skipped,
    }
