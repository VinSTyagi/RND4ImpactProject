from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_IMAGE_SETUP = Path(__file__).resolve().parents[1] / "src" / "image_setup"
if str(_IMAGE_SETUP) not in sys.path:
    sys.path.insert(0, str(_IMAGE_SETUP))

from utils.diffusion_wrapper import resolve_refiner_denoising_start
from utils.config import RefinementConfig, refinement_active, load_config


def test_resolve_refiner_denoising_start_for_latent() -> None:
    ref_cfg = RefinementConfig()
    assert resolve_refiner_denoising_start(object(), ref_cfg) == 0.8


def test_resolve_refiner_denoising_start_for_pil() -> None:
    ref_cfg = RefinementConfig(image_denoising_start=0.01)
    image = Image.new("RGB", (64, 64))
    assert resolve_refiner_denoising_start(image, ref_cfg) == 0.01


def test_image_setup_40gb_refiner_config_loads() -> None:
    config = load_config("configs/image_setup_40gb.yaml")
    assert refinement_active(config)
    assert config.refinement_config.type == "sdxl_refiner"
    assert config.output_config.output_subdir == "refined_images"
    assert (
        config.refinement_config.denoising_start
        == config.refinement_config.denoising_end
    )
