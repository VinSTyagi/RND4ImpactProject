from __future__ import annotations

import sys
from pathlib import Path

_SRC_SCRIPT_SETUP = Path(__file__).resolve().parents[1] / "src" / "script_setup"
if str(_SRC_SCRIPT_SETUP) not in sys.path:
    sys.path.insert(0, str(_SRC_SCRIPT_SETUP))

from utils.stage_2 import parse_title_from_text


def test_parse_title_from_plain_text() -> None:
    assert (
        parse_title_from_text("Pulsing Hellish Sphere Threatens")
        == "Pulsing Hellish Sphere Threatens"
    )


def test_parse_title_from_json_string() -> None:
    assert parse_title_from_text('"Ash Garden"') == "Ash Garden"


def test_parse_title_from_json_object() -> None:
    assert parse_title_from_text('{"title": "Ash Garden"}') == "Ash Garden"


def test_parse_title_strips_title_prefix() -> None:
    assert parse_title_from_text("Title: Ash Garden") == "Ash Garden"
