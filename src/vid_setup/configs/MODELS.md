# Image-to-video model reference (vid_setup)

Configs set `video_diffuser_config.type` to one of the backends below. Use the
`*_low_vram.yaml` presets on ~6GB GPUs; raise resolution/frame count only when
you have headroom.


| Type        | Model (Hugging Face)                                 | Params | ~VRAM (low preset) | Prompt required | Notes                                                   |
| ----------- | ---------------------------------------------------- | ------ | ------------------ | --------------- | ------------------------------------------------------- |
| `svd`       | `stabilityai/stable-video-diffusion-img2vid`         | ~1.5B  | 6GB                | No              | Best no-prompt option; 14 frames; UNet forward chunking |
| `ltx`       | `oxide-lab/LTX-Video-0.9.8-2B-distilled`             | 2B     | 6GB                | Yes             | Fast distilled steps; group offload                     |
| `sana`      | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` | 2B     | 6–8GB              | Yes             | Efficient DiT; uses `frames` kwarg internally           |
| `cogvideox` | `THUDM/CogVideoX-5b-I2V`                             | 5B     | 6–8GB*             | Yes             | Fixed 720×480 native; sequential offload is slow        |
| `wan`       | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`               | 14B    | 8GB+               | Yes             | Highest quality here; no 1.3B I2V checkpoint            |


 CogVideoX reports ~4GB with all diffusers optimizations enabled on A100; expect
higher peak on 6GB consumer cards.

## Config files


| Config                              | Backend          | Target               |
| ----------------------------------- | ---------------- | -------------------- |
| `vid_setup_svd_low_vram.yaml`       | SVD              | ~6GB, no text prompt |
| `vid_setup_ltx_low_vram.yaml`       | LTX 2B distilled | ~6GB                 |
| `vid_setup_sana_low_vram.yaml`      | SANA-Video 2B    | ~6GB                 |
| `vid_setup_cogvideox_low_vram.yaml` | CogVideoX 5B I2V | ~6GB (slow)          |
| `vid_setup_wan_low_vram.yaml`       | Wan 2.1 I2V 14B  | ~8GB+                |
| `vid_setup_svd.yaml`                | SVD              | ~8GB at 1024×576     |
| `vid_setup_ltx.yaml`                | LTX 2B           | ~8GB at 768×432      |


## Prompts

Prompt-based backends (`ltx`, `sana`, `cogvideox`, `wan`) load `data/<script_id>/script.json`
via `[utils/script.py](../utils/script.py)` and use each scene's `image_prompt.positive_prompt`
and `image_prompt.negative_prompt` (from script_setup stage 4). Config
`generation_config.prompt` / `negative_prompt` are used when a scene has no `image_prompt`.

## Tuning ladder (when OOM)

1. Lower `generation_config.width` / `height` (e.g. 512×288 → 416×240)
2. Reduce `num_frames` (SVD: 14 max; others: try 9–17)
3. Enable `enable_sequential_cpu_offload: true` (already on in low-VRAM configs)
4. For LTX only: `enable_group_offload: true` instead of sequential offload
5. For SVD: keep `unet_enable_forward_chunking: true` and `decode_chunk_size: 1`

## Not included (heavier or different workflow)

- **Wan 2.1 T2V 1.3B** — text-to-video only (no image conditioning in diffusers I2V)
- **CogVideoX-2b** — text-to-video only; use `CogVideoX-5b-I2V` for image-to-video
- **LTX 13B / HunyuanVideo / Mochi** — 16GB+ even with offload
-  — SD 1.5 motion adapter (different pipeline; not wired here)

