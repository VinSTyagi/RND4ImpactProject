from __future__ import annotations

import sys
from pathlib import Path

_SRC_SCRIPT_SETUP = Path(__file__).resolve().parents[1] / "src" / "script_setup"
if str(_SRC_SCRIPT_SETUP) not in sys.path:
    sys.path.insert(0, str(_SRC_SCRIPT_SETUP))

from utils.image_prompt import coerce_line_indices


def test_coerce_line_indices_accepts_integers() -> None:
    assert coerce_line_indices([0, 2], beat_count=5) == [0, 2]


def test_coerce_line_indices_accepts_index_objects() -> None:
    assert coerce_line_indices([{"index": 1}], beat_count=3) == [1]


def test_coerce_line_indices_rejects_out_of_range() -> None:
    try:
        coerce_line_indices([5], beat_count=3)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_coerce_line_indices_rejects_non_integers() -> None:
    try:
        coerce_line_indices([["Narration", "text"]], beat_count=3)
    except ValueError as exc:
        assert "must be an integer" in str(exc)
    else:
        raise AssertionError("expected ValueError")
