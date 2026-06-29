from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

_VID_SETUP = Path(__file__).resolve().parents[1] / "src" / "vid_setup"
if str(_VID_SETUP) not in sys.path:
    sys.path.insert(0, str(_VID_SETUP))

from utils.schema import SceneScript, validate_scripts_for_video


def _scene_script(image_prompt: object, *, scene_number: int = 0) -> SceneScript:
    return SceneScript(
        script_id=UUID("9f3b60f2-d1ea-4e39-9773-6e0614761995"),
        scene={"scene_number": scene_number},
        image_prompt=image_prompt,  # type: ignore[arg-type]
    )


def test_scene_prompts_reads_image_prompt_tags() -> None:
    scene_script = _scene_script(
        [
            {
                "positive_prompt": ["hero on cliff", "wide shot"],
                "negative_prompt": ["blurry", "text"],
            }
        ]
    )
    positive, negative = scene_script.scene_prompts(0, 0)
    assert positive == "hero on cliff, wide shot"
    assert negative == "blurry, text"


def test_scene_prompts_does_not_use_config_fallback() -> None:
    scene_script = _scene_script(None)
    positive, negative = scene_script.scene_prompts(0, 0)
    assert positive == ""
    assert negative == ""


def test_validate_scripts_for_video_requires_image_prompt_positive() -> None:
    script_id = "9f3b60f2-d1ea-4e39-9773-6e0614761995"
    scene_scripts = {
        script_id: [
            _scene_script(
                [{"positive_prompt": [], "negative_prompt": ["blurry"]}],
                scene_number=0,
            )
        ]
    }
    try:
        validate_scripts_for_video(
            __import__("logging").getLogger("test"),
            scene_scripts,
            {script_id: 1},
        )
    except ValueError as exc:
        assert "image_prompt.positive_prompt" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing positive prompt")
