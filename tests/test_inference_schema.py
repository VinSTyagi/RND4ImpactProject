from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

_SRC_SCRIPT_SETUP = Path(__file__).resolve().parents[1] / "src" / "script_setup"
if str(_SRC_SCRIPT_SETUP) not in sys.path:
    sys.path.insert(0, str(_SRC_SCRIPT_SETUP))

if "PIL" not in sys.modules:
    sys.modules["PIL"] = types.ModuleType("PIL")
    sys.modules["PIL.Image"] = types.ModuleType("Image")

from utils.image_prompt import coerce_line_indices
from utils.schema import (
    SceneScript,
    StoryIdea,
    coerce_int_field,
    normalize_act,
)

_SCENE_CONTENT = [
    ("Narration", "The bridge creaks like a dying whale under the storm."),
    ("Mara Voss", "We need to cross before dawn."),
]


def test_normalize_act_accepts_common_variants() -> None:
    assert normalize_act("Setup") == "setup"
    assert normalize_act("rising action") == "rising_action"
    assert normalize_act("falling-action") == "falling_action"


def test_coerce_int_field_accepts_string_numbers() -> None:
    assert coerce_int_field("3", "scene_number") == 3
    assert coerce_int_field(2.0, "scene_number") == 2


def test_parse_scene_dict_coerces_llm_variants() -> None:
    scene = SceneScript.parse_scene_dict(
        {
            "scene_number": "0",
            "scene_title": "Ash on the Threshold",
            "act": "Rising Action",
            "setting": "Dawn in a flooded pharmacy.",
            "characters": "Mara Voss",
            "summary": "Mara trades forged papers for medicine while an enforcer watches.",
            "conflict": "She must stay unseen.",
            "emotional_beat": "tense dread",
            "character_change": "Fear replaces control.",
            "ends_on": "The ledger gains a new line.",
        }
    )
    assert scene["scene_number"] == 0
    assert scene["act"] == "rising_action"
    assert scene["characters"] == ["Mara Voss"]


def test_coerce_line_indices_validates_against_beat_count() -> None:
    assert coerce_line_indices([0, 1], beat_count=2) == [0, 1]


def test_scene_script_round_trip_matches_schema() -> None:
    script_id = uuid4()
    with patch("utils.image_prompt.clip_token_count", return_value=1):
        image_prompt = SceneScript.parse_img_prompt_dict(
            {
                "positive_prompt": "storm bridge, lone woman, cinematic",
                "negative_prompt": "blurry, low quality",
                "style_preset": "cinematic",
                "aspect_ratio": "16:9",
                "cfg_scale": 7,
                "reasoning": "Wide shot sells the peril.",
                "lines_used": [0],
            },
            beat_count=2,
            require_lines_used=True,
        )
    scene_script = SceneScript(
        script_id=script_id,
        model="test-model",
        scene=SceneScript.parse_scene_dict(
            {
                "scene_number": 0,
                "scene_title": "Crossing",
                "act": "setup",
                "setting": "A storm-lashed bridge at night.",
                "characters": ["Mara Voss"],
                "summary": "Mara weighs whether to cross the failing bridge.",
                "conflict": "Time is running out.",
                "emotional_beat": "fearful resolve",
                "character_change": "She commits to the crossing.",
                "ends_on": "She steps onto the first plank.",
            }
        ),
        scene_content=_SCENE_CONTENT,
        image_prompt=[image_prompt],
    )
    payload = scene_script.to_json()
    with patch("utils.image_prompt.clip_token_count", return_value=1):
        reloaded = SceneScript.from_dict(payload)
    assert reloaded.scene_content == _SCENE_CONTENT
    assert reloaded.image_prompt is not None
    assert reloaded.image_prompt[0]["lines_used"] == [0]
    assert "scene_content" not in reloaded.scene


def test_story_idea_round_trip_matches_schema() -> None:
    idea = StoryIdea.from_idea_dict(
        {
            "genre": "thriller",
            "setting": "A coastal town in winter.",
            "premise": "A medic races against curfew.",
            "characters": {
                "Mara Voss": "Female, early 30s, alto with clipped urgency — breath short when lying, steadies when treating wounds. Field medic, stubborn, trades forged papers for medicine.",
                "Enforcer Hale": "Male, 40s, low baritone, flat and unhurried — consonants land like stamps, no warmth in vowels. Occupation enforcer who catalogs faces in a ledger.",
            },
            "hook": "Forged papers hidden in a hymn book.",
            "tone": "tense, cold, urgent",
            "theme": "Survival demands moral compromise.",
        },
        model="test-model",
    )
    idea.title = "Ash Ledger"
    payload = idea.to_json()
    reloaded = StoryIdea.from_dict(payload)
    assert reloaded.title == "Ash Ledger"
    assert len(reloaded.characters) == 2
    assert reloaded.characters["Mara Voss"].startswith("Female")
    assert json.loads(json.dumps(payload)) == payload
