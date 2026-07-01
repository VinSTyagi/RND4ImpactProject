from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

_AUDIO_SETUP = Path(__file__).resolve().parents[1] / "src" / "audio_setup"
if str(_AUDIO_SETUP) not in sys.path:
    sys.path.insert(0, str(_AUDIO_SETUP))

if "qwen_tts" not in sys.modules:
    qwen_tts = types.ModuleType("qwen_tts")

    class _Qwen3TTSModel:
        pass

    qwen_tts.Qwen3TTSModel = _Qwen3TTSModel
    sys.modules["qwen_tts"] = qwen_tts

from utils.config import TTSModelConfig, scene_audio_output_path
from utils.schema import SceneScript, StoryIdea
from utils.tts_wrapper import (
    generate_voice_scene,
    save_voices,
    validate_scene_for_audio,
    voiceable_beats,
)

_LOGGER = logging.getLogger("test_audio_setup")

_IDEA_FIELDS = {
    "genre": "thriller",
    "setting": "A coastal town in winter.",
    "premise": "A medic races against curfew.",
    "hook": "Forged papers hidden in a hymn book.",
    "tone": "tense, cold, urgent",
    "theme": "Survival demands moral compromise.",
}


def _write_idea(story_dir: Path, script_id: str, characters: dict[str, str]) -> None:
    story_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "script_id": script_id,
        **_IDEA_FIELDS,
        "characters": characters,
        "title": "Ash Ledger",
    }
    (story_dir / "idea.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_scene_script(
    scene_dir: Path,
    *,
    script_id: str,
    scene_number: int,
    scene_content: list[list[str]],
) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "script_id": script_id,
        "model": "test-model",
        "scene": {
            "scene_number": scene_number,
            "scene_title": f"Scene {scene_number}",
            "act": "setup",
            "setting": "A storm-lashed bridge at night.",
            "characters": ["Mara Voss"],
            "summary": "Mara weighs whether to cross.",
            "conflict": "Time is running out.",
            "emotional_beat": "fearful resolve",
            "character_change": "She commits.",
            "ends_on": "She steps forward.",
        },
        "scene_content": scene_content,
    }
    (scene_dir / "script.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def story_tree(tmp_path: Path) -> tuple[Path, str]:
    script_id = str(uuid4())
    story_dir = tmp_path / script_id
    characters = {
        "Mara Voss": "Female, early 30s, alto with clipped urgency.",
        "Narration": "Neutral narrator, warm baritone, measured pace.",
    }
    _write_idea(story_dir, script_id, characters)
    _write_scene_script(
        story_dir / "0",
        script_id=script_id,
        scene_number=0,
        scene_content=[
            ["Narration", "The bridge creaks in the storm."],
            ["Mara Voss", "We need to cross before dawn."],
            ["(silence)", ""],
        ],
    )
    _write_scene_script(
        story_dir / "1",
        script_id=script_id,
        scene_number=1,
        scene_content=[
            ["Mara Voss", "No turning back now."],
        ],
    )
    return tmp_path, script_id


def test_read_all_loads_every_scene(story_tree: tuple[Path, str]) -> None:
    data_root, script_id = story_tree
    scripts = SceneScript.read_all(str(data_root))
    assert len(scripts) == 2
    assert {int(s.scene["scene_number"]) for s in scripts} == {0, 1}
    assert all(str(s.script_id) == script_id for s in scripts)


def test_voiceable_beats_cover_cast_dialogue(story_tree: tuple[Path, str]) -> None:
    data_root, script_id = story_tree
    idea = StoryIdea.load(str(data_root), script_id)
    scripts = SceneScript.read_all(str(data_root))

    scene_0 = next(s for s in scripts if s.scene["scene_number"] == 0)
    beats = voiceable_beats(scene_0, idea)
    assert beats == [
        (0, "Narration", "The bridge creaks in the storm."),
        (1, "Mara Voss", "We need to cross before dawn."),
    ]

    scene_1 = next(s for s in scripts if s.scene["scene_number"] == 1)
    assert voiceable_beats(scene_1, idea) == [(0, "Mara Voss", "No turning back now.")]


def test_validate_scene_for_audio_warns_on_missing_profiles(
    story_tree: tuple[Path, str],
) -> None:
    data_root, script_id = story_tree
    idea = StoryIdea.load(str(data_root), script_id)
    script = SceneScript.read_for_story(str(data_root), script_id)[0]

    extra = SceneScript.from_dict(
        {
            **json.loads((data_root / script_id / "0" / "script.json").read_text()),
            "scene_content": [
                ["Narration", "Wind howls."],
                ["Unknown Guard", "Halt."],
            ],
        }
    )
    warnings = validate_scene_for_audio(extra, idea)
    assert any("Unknown Guard" in item for item in warnings)


def test_generate_and_save_writes_one_wav_per_voiceable_beat(
    story_tree: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, script_id = story_tree
    idea = StoryIdea.load(str(data_root), script_id)
    scripts = SceneScript.read_all(str(data_root))

    class _FakeModel:
        def generate_voice_design(self, text: str, language: str, instruct: str):
            sample = np.zeros(1600, dtype=np.float32)
            return [sample], 16000

    config = TTSModelConfig(
        input_subdir=str(data_root),
        output_subdir="audio",
        skip_existing=False,
    )

    expected_paths: list[Path] = []
    for script in scripts:
        clips = generate_voice_scene(_LOGGER, _FakeModel(), script, idea)
        save_voices(
            _LOGGER,
            clips,
            script_id=script_id,
            config=config,
        )
        for clip in clips:
            expected_paths.append(
                scene_audio_output_path(
                    script_id,
                    clip["scene_number"],
                    clip["line_index"],
                    config,
                )
            )

    assert len(expected_paths) == 3
    assert all(path.is_file() for path in expected_paths)
    assert expected_paths[0].name == "scene_00_00.wav"
    assert expected_paths[1].name == "scene_00_01.wav"
    assert expected_paths[2].name == "scene_01_00.wav"


def test_generate_voice_scene_rejects_script_idea_mismatch(
    story_tree: tuple[Path, str],
) -> None:
    data_root, script_id = story_tree
    idea = StoryIdea.load(str(data_root), script_id)
    script = SceneScript.read_for_story(str(data_root), script_id)[0]
    mismatched = SceneScript(
        script_id=uuid4(),
        model=script.model,
        scene=script.scene,
        scene_content=script.scene_content,
    )

    with pytest.raises(ValueError, match="does not match idea.json"):
        validate_scene_for_audio(mismatched, idea)
