import contextlib

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

from utils.schema import (
    GenerationConfig,
    QuantizationConfig,
    UpscaleConfig,
    VideoDiffuserConfig,
)


@contextlib.contextmanager
def start_vid_diff_engine(init_config: VideoDiffuserConfig, quant_config: QuantizationConfig):
    """
    Uses the configs to generate

    Args:
        pipeline (VideoDiffuserConfig): pipeline needed to generate the engine
        quant_config (QuantizationConfig): quantization configuration
    """
    try:
        pipe = DiffusionPipeline(init_config.model_path, init_config.torch_dtype)
        if init_config.device and init_config.device != "cpu":
            pipe.to(init_config.device)
        if quant_config.enable_attention_slicing:
            pipe.enable_attention_slicing()
        if quant_config.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        if quant_config.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        if quant_config.enable_vae_slicing:
            pipe.enable_vae_slicing()
        if quant_config.enable_vae_tiling:
            pipe.enable_vae_tiling()
        if quant_config.enable_xformers:
            pipe.enable_xformers()
        yield pipe
    finally:
        del pipe
        return
    
    
def start_upscale_engine(init_config: UpscaleConfig):
    pass


def generate_video_from_images(
    pipe: DiffusionPipeline,
    gen_config: GenerationConfig,
    input_path: str,
    output_path: str,
):
    """Generates a video from a given input path and output path

    Args:
        pipe (DiffusionPipeline): The pipeline to use for generation
        gen_config (GenerationConfig): The generation configuration
        input_path (str): The path to the input image
        output_path (str): The path to the output video

    Raises:
        ValueError: If the image cannot be opened or the video cannot be generated

    Returns:
        str: The path to the output video
    """
    try:
        image = Image.open(input_path)
    except Exception as e:
        raise ValueError(f"Error opening image: {e}")
    
    try:
        args = {k: v for k, v in gen_config.__dict__.items() if v is not None}
        frames = pipe(image=image, **args)
        export_fps = (
            gen_config.fps
            if gen_config.fps is not None
            else gen_config.frame_rate
        )
        if export_fps is None:
            raise ValueError(
                "generation_config must set fps (svd) or frame_rate (ltx) for export"
            )
        export_to_video(frames, output_path, fps=export_fps)
        return output_path
    except Exception as e:
        raise ValueError(f"Error generating video: {e}")
    
    
def upscale_video(pipe: DiffusionPipeline, gen_config: GenerationConfig, input_path: str, output_path: str):
    pass