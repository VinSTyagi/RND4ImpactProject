from __future__ import annotations

import logging
import time

from tqdm import tqdm

from utils.schema import VidSetupPipelineConfig, scene_paths_by_script
from utils.vid_diffuser_wrapper import generate_video_from_images, start_vid_diff_engine


def run_stage(
    logger: logging.Logger,
    config: VidSetupPipelineConfig,
    state: dict,
) -> dict:
    io_cfg = config.io_config
    paths_by_script = scene_paths_by_script(io_cfg)
    if not paths_by_script:
        logger.warning(
            "No scripts with scene images found under %s", io_cfg.script_path
        )
        return {"scripts": 0, "raw_written": 0, "raw_skipped": 0}

    written = 0
    skipped = 0
    start = time.perf_counter()

    with start_vid_diff_engine(
        config.video_diffuser_config,
        config.quantization_config,
    ) as pipeline:
        for script_id, scenes in paths_by_script.items():
            for scene_number, scene in tqdm(
                enumerate(scenes),
                desc=f"Generating videos for script {script_id}",
                total=len(scenes),
            ):
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
                generate_video_from_images(
                    pipeline,
                    config.generation_config,
                    str(scene.image),
                    str(scene.raw_video),
                )
                logger.info(
                    "Generated video for script %s scene %s -> %s",
                    script_id,
                    scene_number,
                    scene.raw_video,
                )
                written += 1

    elapsed = time.perf_counter() - start
    logger.info(
        "Stage 1 complete: %s script(s), %s video(s) written, %s skipped in %.2fs",
        len(paths_by_script),
        written,
        skipped,
        elapsed,
    )
    return {
        "scripts": len(paths_by_script),
        "raw_written": written,
        "raw_skipped": skipped,
    }
