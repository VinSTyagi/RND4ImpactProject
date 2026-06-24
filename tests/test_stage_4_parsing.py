from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_SRC_SCRIPT_SETUP = Path(__file__).resolve().parents[1] / "src" / "script_setup"
if str(_SRC_SCRIPT_SETUP) not in sys.path:
    sys.path.insert(0, str(_SRC_SCRIPT_SETUP))

# schema imports PIL for image types; stub it for lightweight unit tests.
if "PIL" not in sys.modules:
    sys.modules["PIL"] = types.ModuleType("PIL")
    sys.modules["PIL.Image"] = types.ModuleType("Image")

from utils.llm_helper import strip_reasoning
from utils.stage_4 import parse_content_from_text, validate_beat_count


_STAGE_4_EXAMPLE = """{
  "scene_content": [
    ["Narration", "Rain needles the pharmacy windows."],
    ["Mara Voss", "I'm here for antibiotics. Tonight."],
    ["(inner thought)", "His ledger is open."],
    ["Mara Voss", ""]
  ]
}"""


def test_parse_wrapped_scene_content_object() -> None:
    content = parse_content_from_text(_STAGE_4_EXAMPLE)
    assert len(content) == 4
    assert content[0] == ("Narration", "Rain needles the pharmacy windows.")
    assert content[3] == ("Mara Voss", "")


def test_strip_reasoning_preserves_wrapped_object() -> None:
    stripped = strip_reasoning(f"analysis text\n{_STAGE_4_EXAMPLE}")
    content = parse_content_from_text(stripped)
    assert len(content) == 4


def test_parse_bare_scene_content_array() -> None:
    bare = """[
      ["Narration", "Wind howls."],
      ["Dr. Elara Voss", "Not tonight."]
    ]"""
    content = parse_content_from_text(bare)
    assert content == [
        ("Narration", "Wind howls."),
        ("Dr. Elara Voss", "Not tonight."),
    ]


def test_parse_object_form_pairs() -> None:
    payload = """{
      "scene_content": [
        {"character": "Narration", "line": "Dawn breaks."},
        {"speaker": "Mara", "text": "We move."}
      ]
    }"""
    content = parse_content_from_text(payload)
    assert content == [("Narration", "Dawn breaks."), ("Mara", "We move.")]


def test_parse_unescaped_interior_quotes() -> None:
    broken = """{
  "scene_content": [
    ["Narration", "She said "hello" quietly."],
    ["Mara Voss", "Fine."]
  ]
}"""
    content = parse_content_from_text(broken)
    assert len(content) == 2
    assert "hello" in content[0][1]


def test_validate_beat_count() -> None:
    content = [("Narration", "x")] * 20
    validate_beat_count(content, min_beats=18, max_beats=40)
    with pytest.raises(ValueError, match="expected 18-40"):
        validate_beat_count(content[:10], min_beats=18, max_beats=40)


def test_missing_scene_content_field_raises() -> None:
    with pytest.raises(ValueError, match="missing field: scene_content"):
        parse_content_from_text('{"dialogue": []}')
