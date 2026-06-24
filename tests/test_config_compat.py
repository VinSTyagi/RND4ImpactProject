from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pipeline_validate
from pipeline_validate import validate_bundle


def test_all_stage_configs_load() -> None:
    pipeline_validate.validate_all_setup_configs()


def test_all_pipeline_profiles_validate() -> None:
    pipeline_validate.validate_all_pipeline_profiles()


@pytest.mark.parametrize(
    ("script_config", "image_config", "vid_config"),
    [
        (
            "configs/script_setup_qwen3_4b.yaml",
            "configs/image_setup_sdxl_fp16.yaml",
            "configs/vid_setup_svd.yaml",
        ),
        (
            "configs/script_setup_40gb.yaml",
            "configs/image_setup_40gb.yaml",
            "configs/vid_setup_40gb.yaml",
        ),
    ],
)
def test_known_bundles_validate(
    script_config: str,
    image_config: str,
    vid_config: str,
) -> None:
    validate_bundle(script_config, image_config, vid_config)
