from __future__ import annotations

import logging
import time

import torch
from tqdm import tqdm

from utils.config import (
    VidSetupPipelineConfig,
    pipeline_needs_prompt,
    scene_paths_by_script,
)
from utils.schema import SceneScript, validate_scripts_for_video
from utils.vid_diffuser_wrapper import generate_video_from_images, start_vid_diff_engine


def _scene_scripts_for_paths(
    io_cfg,
    paths_by_script: dict[str, list],
) -> dict[str, list[SceneScript]]:
    """Load per-scene script.json for every story with scene images."""
    if not paths_by_script:
        return {}

    scene_scripts_by_id: dict[str, list[SceneScript]] = {}
    for scene_script in SceneScript.read_all(io_cfg.script_path):
        script_id = str(scene_script.script_id)
        if script_id not in paths_by_script:
            continue
        scene_scripts_by_id.setdefault(script_id, []).append(scene_script)

    for script_id in scene_scripts_by_id:
        scene_scripts_by_id[script_id].sort(key=lambda item: item.scene_number())

    missing = sorted(set(paths_by_script) - set(scene_scripts_by_id))
    if missing:
        raise FileNotFoundError(
            "per-scene script.json missing for script(s) with scene images: "
            + ", ".join(missing)
        )
    return scene_scripts_by_id


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

    scene_scripts_by_id = _scene_scripts_for_paths(io_cfg, paths_by_script)
    logger.info(
        "Loaded scene scripts for %s story(ies) from %s",
        len(scene_scripts_by_id),
        io_cfg.script_path,
    )

    needs_prompt = pipeline_needs_prompt(pipeline_type)
    if needs_prompt:
        prompt_counts = {
            script_id: len(scenes) for script_id, scenes in paths_by_script.items()
        }
        validate_scripts_for_video(logger, scene_scripts_by_id, prompt_counts)

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
        for script_id, scene_paths in paths_by_script.items():
            scene_scripts = scene_scripts_by_id[script_id]
            scene_scripts_by_number = {
                scene_script.scene_number(): scene_script
                for scene_script in scene_scripts
            }
            for scene in tqdm(
                scene_paths,
                desc=f"Generating videos for script {script_id}",
                total=len(scene_paths),
            ):
                scene_number = scene.scene_number
                prompt_number = scene.prompt_number
                if io_cfg.skip_existing and scene.raw_video.is_file():
                    logger.info(
                        "Skipping script %s scene %s prompt %s (output exists): %s",
                        script_id,
                        scene_number,
                        prompt_number,
                        scene.raw_video,
                    )
                    skipped += 1
                    continue

                scene.raw_video.parent.mkdir(parents=True, exist_ok=True)
                prompt = None
                negative_prompt = None
                if needs_prompt:
                    scene_script = scene_scripts_by_number.get(scene_number)
                    if scene_script is None:
                        raise ValueError(
                            f"{script_id}: no script.json for scene {scene_number}"
                        )
                    prompt, negative_prompt = scene_script.scene_prompts(
                        scene_number,
                        prompt_number,
                    )

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
                        "Failed script %s scene %s prompt %s: %s",
                        script_id,
                        scene_number,
                        prompt_number,
                        exc,
                    )
                    failed += 1
                    if using_offload and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                logger.info(
                    "Generated video for script %s scene %s prompt %s -> %s",
                    script_id,
                    scene_number,
                    prompt_number,
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
