from __future__ import annotations

import logging
from typing import Iterator

from utils import diffusion_wrapper
from utils.schema import (
    ImageSetupPipelineConfig,
    Scene,
    Script,
    scene_output_path,
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


def run_stage(
    logger: logging.Logger,
    config: ImageSetupPipelineConfig,
    state: dict,
) -> dict:
    output_cfg = config.output_config
    gen_cfg = config.generation_config
    pipeline_type = config.pipeline_config.type.strip().lower()

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
                output_path = scene_output_path(
                    script.script_id, scene_number, output_cfg
                )

                if output_cfg.skip_existing and output_path.is_file():
                    logger.info(
                        "Skipping script %s scene %s (output exists): %s",
                        script_id,
                        scene_number,
                        output_path,
                    )
                    skipped += 1
                    continue

                result = diffusion_wrapper.generate_scene_image(
                    pipeline,
                    scene["image_prompt"],
                    gen_cfg,
                    pipeline_type=pipeline_type,
                    pipeline_cfg=pipeline_cfg,
                    seed_offset=scene_number,
                )

                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(output_path)
                logger.info(
                    "Saved image for script %s scene %s to %s",
                    script_id,
                    scene_number,
                    output_path,
                )
                written += 1

    state["scripts"] = scripts
    return {
        "scripts": len(scripts),
        "written": written,
        "skipped": skipped,
    }
