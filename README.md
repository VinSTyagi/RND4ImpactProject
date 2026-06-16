# RND4Impact Project

ML environment for image/video diffusion, NLP, and TTS research.

## Prerequisites

- **Python 3.10–3.12** (install scripts default to **3.11**)
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** package manager
- **Git** (for Stability `generative-models` clone)
- **NVIDIA GPU + driver** supporting CUDA 12.4 (default install target)

## Install

From the repository root:

**Windows (PowerShell):**

```powershell
.\scripts\install.ps1
```

**Linux / WSL / macOS:**

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

### Options


| Flag                | PowerShell              | Description             |
| ------------------- | ----------------------- | ----------------------- |
| Skip Stability venv | `-SkipStability`        | `--skip-stability`      |
| CPU-only PyTorch    | `-CpuOnly`              | `--cpu-only`            |
| Python version      | `-PythonVersion 3.11`   | `--python-version 3.11` |
| Jupyter kernel      | `-InstallJupyterKernel` | `--jupyter-kernel`      |
| Optional decord     | `-InstallDecord`        | `--install-decord`      |


Example (main stack only, CPU):

```powershell
.\scripts\install.ps1 -SkipStability -CpuOnly
```

## Run with Docker (GPU)

Three separate Docker Compose setups share only the host `data/` folder (pipeline
artifact handoff). Each stack has its own image, dependency manifest, and Hugging
Face cache volume (`rnd4impact_script_hf_cache` / `rnd4impact_image_hf_cache` /
`rnd4impact_vid_hf_cache`):


| Setup                                          | Image                       | Purpose                           |
| ---------------------------------------------- | --------------------------- | --------------------------------- |
| `[docker/script_setup/](docker/script_setup/)` | `rnd4impact-script:cuda124` | vLLM script pipeline (stages 1–4) |
| `[docker/image_setup/](docker/image_setup/)`   | `rnd4impact-image:cuda124`  | SDXL scene image generation       |
| `[docker/vid_setup/](docker/vid_setup/)`       | `rnd4impact-vid:cuda124`    | Video diffusion (SVD) generation  |


vLLM ships compiled CUDA extensions (`vllm._C`) that are **Linux-only**, so
`script_setup` cannot run on native Windows (`ModuleNotFoundError: No module named 'vllm._C'`). Run it in the script_setup container. **image_setup** reads
`data/<script_id>/script.json` (with `image_prompt` from script_setup stage 4)
and writes PNGs to `data/<script_id>/refined_images/`. **vid_setup** reads those
PNGs and writes scene videos to `data/<script_id>/raw_videos/`.

### Prerequisites

- **Docker Desktop** with the **WSL2 backend** and GPU support enabled
(Settings -> Resources -> WSL integration), or Docker on native Linux.
- A recent **NVIDIA driver** on the host (Windows driver covers WSL2).
- **NVIDIA Container Toolkit** (installed automatically inside Docker Desktop's
WSL2 backend; install manually on native Linux).

Verify GPU passthrough works:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Build

**script_setup** (vLLM stack):

```powershell
.\scripts\docker-build-script-setup.ps1
```

```bash
chmod +x scripts/docker-build-script-setup.sh
./scripts/docker-build-script-setup.sh
```

**image_setup** (diffusers / SDXL stack):

```powershell
.\scripts\docker-build-image-setup.ps1
```

```bash
chmod +x scripts/docker-build-image-setup.sh
./scripts/docker-build-image-setup.sh
```

**vid_setup** (diffusers / SVD stack — xformers, accelerate, video I/O):

```powershell
.\scripts\docker-build-vid-setup.ps1
```

```bash
chmod +x scripts/docker-build-vid-setup.sh
./scripts/docker-build-vid-setup.sh
```

The first build is large (CUDA + ML stack). Each Dockerfile runs
`uv sync --directory src/<setup>` against that setup's own `pyproject.toml` (the
workspace root manifest is **not** copied into images). Dependencies re-resolve on
Linux during the image build. Rebuild after changing a setup `pyproject.toml` pin.

Each pipeline is isolated: its own `pyproject.toml`, source tree, and Docker
image. Cross-pipeline handoff uses `data/<script_id>/` artifacts (`script.json`,
`refined_images/`, `raw_videos/`, etc.).

For local development, sync one workspace package or the whole workspace from the
repo root:

```bash
uv sync --package rnd4impact-script-setup   # vLLM script pipeline only
uv sync --package rnd4impact-image-setup    # diffusers image pipeline only
uv sync --package rnd4impact-vid-setup      # diffusers video pipeline only
# or:
uv sync
```

Each GPU pipeline `pyproject.toml` (`script_setup`, `image_setup`, `vid_setup`) pins
**torch 2.6.0+cu124**, **torchvision**, and **xformers** from the PyTorch CUDA 12.4
index on Linux and Windows (`index-strategy = unsafe-best-match`). Use the project
`.venv` — a system-wide CPU-only `torch` install will not use your GPU.

Verify CUDA after syncing a GPU package (example: `rnd4impact-vid-setup`):

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `2.6.0+cu124` and `True` when an NVIDIA driver is installed.

### script_setup — run pipeline stages

The runner takes one flag per stage; combine them freely or use `--all`. All
stages share one vLLM config (`global_vllm_config` in the YAML).

```bash
cd docker/script_setup
docker compose build
docker compose run --rm script-setup        # --all (default command)
# or: ./scripts/docker-run-script-setup.sh --build

# Stage 1 only: idea generation -> data/<script_id>/script.json
docker compose run --rm script-setup \
  python script_setup/script_setup_runner.py --config configs/script_setup_qwen3_4b.yaml --1

# Stage 2 only: titles from existing data/<script_id>/script.json
docker compose run --rm script-setup \
  python script_setup/script_setup_runner.py --config configs/script_setup_qwen3_4b.yaml --2

# All implemented stages (default `script-setup` command)
docker compose run --rm script-setup \
  python script_setup/script_setup_runner.py --config configs/script_setup_qwen3_4b.yaml --all
```

Stage 4 writes `image_prompt` fields into `script.json` (text prompts for
SDXL, not PNGs). Running `docker compose run --rm script-setup` with no command
override executes `--all`.

**Offline inference and model cache** — copy
`[docker/script_setup/.env.example](docker/script_setup/.env.example)` to
`docker/script_setup/.env`, set `HF_TOKEN` if needed, then:

```bash
cd docker/script_setup
docker compose run --rm prefetch-model
# set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in .env, then:
docker compose run --rm script-setup
```

**Other script_setup services:**

```bash
cd docker/script_setup
docker compose run --rm shell
docker compose up jupyter    # http://localhost:8888
```

### image_setup — generate and refine scene images

Run **after** script_setup stage 4 (or `--all`) has populated `image_prompt`
on every scene in `data/<script_id>/script.json`.

The image_setup runner has two stages (like script_setup):


| Stage | Purpose                                             | Output path                                |
| ----- | --------------------------------------------------- | ------------------------------------------ |
| **1** | Raw SDXL generation                                 | `data/<script_id>/images_raw/scene_XX.png` |
| **2** | Refinement (`sdxl_refiner` or `img2img` per config) | `data/<script_id>/refined_images/scene_XX.png` |


Default Docker command runs `--all` (both stages). Turbo config disables
refinement and writes directly to `images_turbo/`.

Models load from Hugging Face on the first run via diffusers `from_pretrained`
and are cached in the `rnd4impact_image_hf_cache` Docker volume. Re-runs reuse the
cache automatically.

```bash
cd docker/image_setup
docker compose build
docker compose run --rm image-setup        # --all (raw + refine)
# or: ./scripts/docker-run-image-setup.sh --build
```

Run individual stages:

```bash
cd docker/image_setup
# Stage 1 only: raw images -> images_raw/
docker compose run --rm image-setup \
  python image_setup/image_setup_runner.py --config configs/image_setup_sdxl_fp16.yaml --1

# Stage 2 only: refine existing raw images -> refined_images/
docker compose run --rm image-setup \
  python image_setup/image_setup_runner.py --config configs/image_setup_sdxl_fp16.yaml --2
```

Low-VRAM SDXL config (img2img refinement on the base model):

```bash
cd docker/image_setup
docker compose run --rm image-setup python image_setup/image_setup_runner.py --config configs/image_setup_sdxl_low_vram.yaml --all
```

Cheaper SD 1.5 config (~4 GB VRAM, img2img refinement on the same base model):

```bash
cd docker/image_setup
docker compose run --rm image-setup python image_setup/image_setup_runner.py --config configs/image_setup_sd15.yaml --all
```

Set `pipeline_config.type` to `sdxl` or `sd15`. SD 1.5 uses 512-based resolutions and
does not support the SDXL refiner backend.

Copy `[docker/image_setup/.env.example](docker/image_setup/.env.example)` to
`docker/image_setup/.env` for `HF_TOKEN` and offline flags.

### vid_setup — generate scene videos from refined images

Run **after** image_setup has written `scene_XX.png` files under
`data/<script_id>/refined_images/`.

| Stage | Purpose                         | Output path                                  |
| ----- | ------------------------------- | -------------------------------------------- |
| **1** | Image-to-video diffusion (SVD/LTX) | `data/<script_id>/raw_videos/scene_XX.mp4` |

Only stage 1 is implemented today (`--1` or `--all`).

Default Docker command uses
`[configs/vid_setup_svd.yaml](src/vid_setup/configs/vid_setup_svd.yaml)` (Stable Video
Diffusion, fp16, 1024×576, 14 frames). An alternate backend is
`[configs/vid_setup_ltx.yaml](src/vid_setup/configs/vid_setup_ltx.yaml)` (LTX-Video).

Models load from Hugging Face on the first run and are cached in the
`rnd4impact_vid_hf_cache` Docker volume.

```bash
cd docker/vid_setup
docker compose build
docker compose run --rm vid-setup        # --all (default command)
# or: ./scripts/docker-run-vid-setup.sh --build
```

Override the config or pass runner flags:

```bash
cd docker/vid_setup
docker compose run --rm vid-setup \
  python vid_setup/vid_setup_runner.py --config configs/vid_setup_ltx.yaml --all

docker compose run --rm vid-setup \
  python vid_setup/vid_setup_runner.py --config configs/vid_setup_svd.yaml --1
```

**Native Windows / Linux (GPU, no Docker):**

```powershell
uv sync --package rnd4impact-vid-setup
cd src\vid_setup
..\..\.venv\Scripts\python.exe vid_setup_runner.py --config configs\vid_setup_svd.yaml --all
```

```bash
uv sync --package rnd4impact-vid-setup
cd src/vid_setup
../../.venv/bin/python vid_setup_runner.py --config configs/vid_setup_svd.yaml --all
```

Copy `[docker/vid_setup/.env.example](docker/vid_setup/.env.example)` to
`docker/vid_setup/.env` for `HF_TOKEN`, `CUDA_VISIBLE_DEVICES`, and offline flags.

**Shell service:**

```bash
cd docker/vid_setup
docker compose run --rm shell
```

### Performance notes

- Compose services request all GPUs; set `CUDA_VISIBLE_DEVICES` in each setup's
`.env` to pin a specific GPU.
- script_setup uses `shm_size: "8gb"` for vLLM; image_setup uses `2gb`; vid_setup
uses `4gb`.
- Tune `global_vllm_config` for your VRAM, e.g.
`[src/script_setup/configs/script_setup_qwen3_4b.yaml](src/script_setup/configs/script_setup_qwen3_4b.yaml)`.
- script_setup compose sets `VLLM_USE_V1=0` by default.
- Default SVD config (`vid_setup_svd.yaml`) enables CPU offload, VAE
slicing/tiling, attention slicing, xformers, and `decode_chunk_size: 1` to fit
~6–8 GB VRAM. Expect slow per-scene generation on 6 GB cards; lower
`generation_config.width` / `height` for faster runs.

## Virtual environments


| Venv              | Use for                                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `.venv`           | SD 1.5, SDXL, ControlNet, LoRA, xFormers, **SVD**, transformers/BERT/GPT-style NLP, Coqui TTS (Bark + Fairseq models), JupyterLab |
| `.venv-stability` | Official **SV3D** and **SV4D** ([Stability-AI/generative-models](https://github.com/Stability-AI/generative-models))              |


```powershell
.\.venv\Scripts\Activate.ps1
# or
.\.venv-stability\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
# or
source .venv-stability/bin/activate
```

## Stacks (main `.venv`)

- **Stable Diffusion:** `diffusers`, `transformers`, `peft`, `controlnet-aux`, `accelerate`, `safetensors`, `invisible-watermark`
- **Video:** SVD / LTX via diffusers (`src/vid_setup/`); I/O via `imageio`, `av`, `opencv-python`, `einops`, `kornia`
- **NLP:** `transformers`, `tokenizers`, `datasets`, `sentencepiece`
- **TTS:** `coqui-tts[languages]` (Bark + Fairseq TTS through Coqui; not legacy `TTS` on PyPI)
- **Utilities:** `jupyterlab`, `matplotlib`, `huggingface-hub`, `pillow`

Pinned PyTorch (default): **torch 2.6.0 + CUDA 12.4 + xformers 0.0.29.post2 + vLLM 0.8.5** (native Qwen3).

## Model weights

Weights are **not** downloaded by the install script. After install, use `huggingface-cli download` (hints printed at end of install) or your own `data/models/` layout.

## Project layout

```
pyproject.toml              # uv workspace root (dev tooling)
docker/
  script_setup/             # vLLM GPU compose (script pipeline)
  image_setup/              # SDXL GPU compose (scene images)
  vid_setup/                # SVD GPU compose (scene videos)
requirements/
  constraints-main.txt      # Torch/xformers pins for install phase A
  stability-pt2.txt         # Stability generative-models deps (stability venv)
scripts/
  docker-build-script-setup.ps1 / .sh
  docker-run-script-setup.ps1 / .sh
  docker-build-image-setup.ps1 / .sh
  docker-run-image-setup.ps1 / .sh
  docker-build-vid-setup.ps1 / .sh
  docker-run-vid-setup.ps1 / .sh
data/                       # Shared pipeline I/O (bind-mounted in Docker)
vendor/generative-models/   # Cloned by install (gitignored)
src/
  script_setup/             # vLLM script pipeline (+ pyproject.toml)
  image_setup/              # SDXL image generation (+ pyproject.toml)
  vid_setup/                # SVD video generation (+ pyproject.toml)
```

## Optional extras (workspace root)

```powershell
uv sync --extra dev
```

