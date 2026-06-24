from __future__ import annotations

import logging
import time

import torch
from tqdm import tqdm

from utils.schema import (
    VidSetupPipelineConfig,
    pipeline_needs_prompt,
    scene_paths_by_script,
)
from utils.script import Script, validate_scripts_for_video
from utils.vid_diffuser_wrapper import generate_video_from_images, start_vid_diff_engine


def run_stage(
    logger: logging.Logger,
    config: VidSetupPipelineConfig,
    state: dict,
) -> dict:
    io_cfg = config.io_config
    gen_cfg = config.generation_config
    quant_cfg = config.quantization_config
    pipeline_type = config.video_diffuser_config.normalized_type()
    paths_by_script = scene_paths_by_script(io_cfg)
    if not paths_by_script:
        logger.warning(
            "No scripts with scene images found under %s", io_cfg.script_path
        )
        return {"scripts": 0, "raw_written": 0, "raw_skipped": 0, "raw_failed": 0}

    scripts_by_id: dict[str, Script] = {}
    if pipeline_needs_prompt(pipeline_type):
        scene_counts = {
            script_id: [scene.scene_number for scene in scenes]
            for script_id, scenes in paths_by_script.items()
        }
        try:
            scripts_by_id = Script.load_by_ids(list(paths_by_script), io_cfg)
        except FileNotFoundError as exc:
            if (gen_cfg.prompt or "").strip():
                logger.warning(
                    "script.json missing for some scripts (%s); using "
                    "generation_config prompt fallbacks",
                    exc,
                )
            else:
                raise ValueError(
                    f"{exc}. Prompted backends require script.json with "
                    "image_prompt or generation_config.prompt."
                ) from exc
        else:
            validate_scripts_for_video(logger, scripts_by_id, scene_counts, gen_cfg)

    written = 0
    skipped = 0
    failed = 0
    start = time.perf_counter()

    offload_mode = (
        "group"
        if quant_cfg.enable_group_offload
        else "sequential"
        if quant_cfg.enable_sequential_cpu_offload
        else "model"
        if quant_cfg.enable_model_cpu_offload
        else "none"
    )
    logger.info(
        "Stage 1 settings: type=%s resolution=%sx%s frames=%s offload=%s",
        pipeline_type,
        gen_cfg.width,
        gen_cfg.height,
        gen_cfg.num_frames,
        offload_mode,
    )

    with start_vid_diff_engine(
        logger,
        config.video_diffuser_config,
        config.quantization_config,
    ) as (pipeline, using_offload):
        for script_id, scenes in paths_by_script.items():
            script = scripts_by_id.get(script_id)
            for scene in tqdm(
                scenes,
                desc=f"Generating videos for script {script_id}",
                total=len(scenes),
            ):
                scene_number = scene.scene_number
                if io_cfg.skip_existing and scene.raw_video.is_file():
                    logger.info(
                        "Skipping script %s scene %s (output exists): %s",
                        script_id,
                        scene_number,
                        scene.raw_video,
                    )
                    skipped += 1
                    continue

                scene.raw_video.parent.mkdir(parents=True, exist_ok=True)
                prompt = None
                negative_prompt = None
                if pipeline_needs_prompt(pipeline_type):
                    if script is not None:
                        prompt, negative_prompt = script.scene_prompts(
                            scene_number,
                            gen_cfg,
                        )
                    else:
                        prompt = (gen_cfg.prompt or "").strip()
                        negative_prompt = (gen_cfg.negative_prompt or "").strip()

                try:
                    generate_video_from_images(
                        logger,
                        pipeline,
                        gen_cfg,
                        pipeline_type,
                        str(scene.image),
                        str(scene.raw_video),
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        using_offload=using_offload,
                    )
                except ValueError as exc:
                    logger.error(
                        "Failed script %s scene %s: %s",
                        script_id,
                        scene_number,
                        exc,
                    )
                    failed += 1
                    if using_offload and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                logger.info(
                    "Generated video for script %s scene %s -> %s",
                    script_id,
                    scene_number,
                    scene.raw_video,
                )
                written += 1

    elapsed = time.perf_counter() - start
    logger.info(
        "Stage 1 complete: %s script(s), %s video(s) written, %s skipped, "
        "%s failed in %.2fs",
        len(paths_by_script),
        written,
        skipped,
        failed,
        elapsed,
    )
    return {
        "scripts": len(paths_by_script),
        "raw_written": written,
        "raw_skipped": skipped,
        "raw_failed": failed,
    }
