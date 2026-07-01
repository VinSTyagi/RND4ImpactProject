# Generates full-scene audio from scene_content dialogue beats.

from argparse import ArgumentParser
from pathlib import Path

import logging
import os

from audio_setup.utils.config import load_config
from audio_setup.utils.schema import SceneScript, StoryIdea
from audio_setup.utils.tts_wrapper import (
    generate_voice_scene,
    launch_tts_engine,
    save_voices,
    validate_scene_for_audio,
)

_DEFAULT_CONFIG = Path("configs/audio_setup_12gb.yaml")


def load_args_parser() -> ArgumentParser:
    args_parser = ArgumentParser(
        description="Generate full-scene audio from script scene_content",
    )
    args_parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to audio setup config",
    )
    return args_parser


def gen_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"),
        force=True,
    )
    return logging.getLogger(__name__)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logger = gen_logging()
    parser = load_args_parser()
    args = parser.parse_args()
    tts_config = load_config(str(args.config))

    ideas: dict[str, StoryIdea] = {}
    scripts = SceneScript.read_all(tts_config.input_subdir)
    scenes_processed = 0
    clips_written = 0

    with launch_tts_engine(logger, tts_config) as model:
        for script in scripts:
            script_id = str(script.script_id)
            if script_id not in ideas:
                ideas[script_id] = StoryIdea.load(tts_config.input_subdir, script_id)
            idea = ideas[script_id]

            for warning in validate_scene_for_audio(script, idea):
                logger.warning("%s", warning)

            clips = generate_voice_scene(logger, model, script, idea)
            if not clips:
                logger.warning(
                    "No audio clips generated for script %s scene %s",
                    script_id,
                    script.scene.get("scene_number"),
                )
                continue

            save_voices(
                logger,
                clips,
                script_id=script_id,
                config=tts_config,
            )
            scenes_processed += 1
            clips_written += len(clips)

    logger.info(
        "Audio setup complete: %s scene(s), %s clip(s)",
        scenes_processed,
        clips_written,
    )
