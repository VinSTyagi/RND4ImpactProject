from __future__ import annotations

import logging
from typing import Iterator

from utils import diffusion_wrapper
from utils.schema import (
    ImageSetupPipelineConfig,
    Scene,
    Script,
    refinement_active,
    scene_output_path,
    scene_raw_output_path,
    validate_pipeline_config,
)

_PIPELINE_LOADERS = {
    "sdxl": diffusion_wrapper.txt2img_session,
    "sd15": diffusion_wrapper.txt2img_session,
}


def validate_scripts(scripts: list[Script]) -> None:
    """Ensure every script has scenes with image prompts before loading the pipeline."""
    errors: list[str] = []
    for script in scripts:
        if not script.script_scenes:
            errors.append(f"{script.script_id}: missing script_scenes")
            continue
        for scene in script.script_scenes:
            if scene.get("image_prompt") is None:
                errors.append(
                    f"{script.script_id}: scene {scene['scene_number']} missing image_prompt"
                )
    if errors:
        raise ValueError(
            "image generation requires image_prompt on every scene "
            "(run stage 4 of the script pipeline first):\n"
            + "\n".join(f"  - {item}" for item in errors)
            + "\n\nPopulate image_prompt in data/<script_id>/script.json, then re-run image_setup."
        )


def iter_scenes(script: Script) -> Iterator[tuple[Script, Scene]]:
    scenes = script.script_scenes or []
    for scene in sorted(scenes, key=lambda item: item["scene_number"]):
        yield script, scene


def _latent_handoff_enabled(
    config: ImageSetupPipelineConfig,
    stages: list[int],
) -> bool:
    ref_cfg = config.refinement_config
    return (
        refinement_active(config)
        and 2 in stages
        and ref_cfg.type.strip().lower() == "sdxl_refiner"
        and not config.output_config.save_raw
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
    use_latent_handoff = (
        family.supports_latent_handoff
        and _latent_handoff_enabled(config, stages)
    )

    validate_pipeline_config(config)

    session_factory = _PIPELINE_LOADERS.get(pipeline_type)
    if session_factory is None:
        supported = ", ".join(sorted(_PIPELINE_LOADERS))
        raise ValueError(
            f"unsupported pipeline type {pipeline_type!r}; supported: {supported}"
        )

    scripts = Script.read_all(output_cfg.script_path)
    logger.info("Loaded %s scripts from %s", len(scripts), output_cfg.script_path)
    validate_scripts(scripts)

    written = 0
    skipped = 0
    latent_handoffs: dict[tuple[str, int], object] = {}

    with session_factory(
        config.pipeline_config,
        config.quantization_config,
        gen_cfg,
    ) as pipeline:
        pipeline_cfg = config.pipeline_config
        for script in scripts:
            for _, scene in iter_scenes(script):
                scene_number = scene["scene_number"]
                script_id = str(script.script_id)

                if refine:
                    raw_path = scene_raw_output_path(
                        script.script_id, scene_number, output_cfg
                    )
                    final_path = scene_output_path(
                        script.script_id, scene_number, output_cfg
                    )
                    if output_cfg.skip_existing:
                        if final_path.is_file():
                            logger.info(
                                "Skipping script %s scene %s (final exists): %s",
                                script_id,
                                scene_number,
                                final_path,
                            )
                            skipped += 1
                            continue
                        if (
                            not use_latent_handoff
                            and raw_path.is_file()
                        ):
                            logger.info(
                                "Skipping script %s scene %s (raw exists): %s",
                                script_id,
                                scene_number,
                                raw_path,
                            )
                            skipped += 1
                            continue
                else:
                    raw_path = scene_output_path(
                        script.script_id, scene_number, output_cfg
                    )
                    if output_cfg.skip_existing and raw_path.is_file():
                        logger.info(
                            "Skipping script %s scene %s (output exists): %s",
                            script_id,
                            scene_number,
                            raw_path,
                        )
                        skipped += 1
                        continue

                denoising_end = (
                    ref_cfg.denoising_end if use_latent_handoff else None
                )
                output_type = "latent" if use_latent_handoff else "pil"

                result = diffusion_wrapper.generate_scene_image(
                    pipeline,
                    scene["image_prompt"],
                    gen_cfg,
                    pipeline_type=pipeline_type,
                    pipeline_cfg=pipeline_cfg,
                    seed_offset=scene_number,
                    output_type=output_type,
                    denoising_end=denoising_end,
                )

                if use_latent_handoff:
                    latent_handoffs[(script_id, scene_number)] = result
                    logger.info(
                        "Generated raw latent for script %s scene %s",
                        script_id,
                        scene_number,
                    )
                else:
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    result.save(raw_path)
                    logger.info(
                        "Saved raw image for script %s scene %s to %s",
                        script_id,
                        scene_number,
                        raw_path,
                    )
                written += 1

    state["scripts"] = scripts
    state["latent_handoffs"] = latent_handoffs
    return {
        "scripts": len(scripts),
        "raw_written": written,
        "raw_skipped": skipped,
    }
