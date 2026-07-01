from __future__ import annotations

import contextlib
import logging
from typing import Any

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from utils.config import TTSModelConfig, scene_audio_output_path
from utils.schema import SceneScript, StoryIdea


@contextlib.contextmanager
def launch_tts_engine(logger: logging.Logger, config: TTSModelConfig):
    logger.info("Launching TTS engine with config: %s", config)
    if config.model_family == "Qwen":
        model = Qwen3TTSModel.from_pretrained(
            config.model_name,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=config.dtype,
        )
    else:
        model = None

    try:
        yield model
    finally:
        del model


def _should_voice_line(
    character: str,
    line_text: str,
    voice_profiles: dict[str, str],
) -> bool:
    return bool(line_text.strip()) and character in voice_profiles


def voiceable_beats(
    script: SceneScript,
    idea: StoryIdea,
) -> list[tuple[int, str, str]]:
    """Return (line_index, character, text) for each beat that will be voiced."""
    profiles = idea.characters
    return [
        (line_index, character, line_text)
        for line_index, (character, line_text) in enumerate(script.scene_content)
        if _should_voice_line(character, line_text, profiles)
    ]


def validate_scene_for_audio(script: SceneScript, idea: StoryIdea) -> list[str]:
    """Return warnings when a scene may not produce audio as expected."""
    warnings: list[str] = []
    scene_number = int(script.scene.get("scene_number", 0))

    if script.script_id != idea.script_id:
        raise ValueError(
            f"scene {scene_number}: script_id {script.script_id} "
            f"does not match idea.json {idea.script_id}"
        )
    if not script.scene_content:
        warnings.append(f"scene {scene_number}: scene_content is empty")

    beats = voiceable_beats(script, idea)
    if not beats:
        warnings.append(
            f"scene {scene_number}: no voiceable lines "
            "(add character profiles in idea.json or dialogue in scene_content)"
        )

    missing_profiles = sorted(
        {
            character
            for character, line_text in script.scene_content
            if line_text.strip() and character not in idea.characters
        }
    )
    if missing_profiles:
        warnings.append(
            f"scene {scene_number}: missing voice profiles for: "
            + ", ".join(missing_profiles)
        )
    return warnings


def generate_voice_scene(
    logger: logging.Logger,
    model,
    script: SceneScript,
    idea: StoryIdea,
) -> list[dict[str, Any]]:
    """Generate voice for every speakable line in scene_content (full-scene audio)."""
    scene_number = int(script.scene.get("scene_number", 0))
    logger.info(
        "Generating full-scene voice for script %s scene %s (%s beat(s))",
        script.script_id,
        scene_number,
        len(script.scene_content),
    )

    voice_profiles = idea.characters
    beats = voiceable_beats(script, idea)
    clips: list[dict[str, Any]] = []

    for line_index, character, line_text in beats:
        logger.info("Beat %s: %s — %s", line_index, character, line_text)
        ref_wavs, sr = model.generate_voice_design(
            text=line_text,
            language="English",
            instruct=voice_profiles[character],
        )
        clips.append(
            {
                "scene_number": scene_number,
                "line_index": line_index,
                "character": character,
                "text": line_text,
                "voice": ref_wavs[0],
                "sr": sr,
            }
        )
    return clips


def save_voices(
    logger: logging.Logger,
    clips: list[dict[str, Any]],
    *,
    script_id: str,
    config: TTSModelConfig,
) -> None:
    for clip in clips:
        path = scene_audio_output_path(
            script_id,
            clip["scene_number"],
            clip["line_index"],
            config,
        )
        if config.skip_existing and path.is_file():
            logger.info("Skipping existing audio: %s", path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), clip["voice"], clip["sr"])
        logger.info("Voice saved: %s", path)
