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

| Flag | PowerShell | Description |
|------|------------|-------------|
| Skip Stability venv | `-SkipStability` | `--skip-stability` |
| CPU-only PyTorch | `-CpuOnly` | `--cpu-only` |
| Python version | `-PythonVersion 3.11` | `--python-version 3.11` |
| Jupyter kernel | `-InstallJupyterKernel` | `--jupyter-kernel` |
| Optional decord | `-InstallDecord` | `--install-decord` |

Example (main stack only, CPU):

```powershell
.\scripts\install.ps1 -SkipStability -CpuOnly
```

## Run with Docker (GPU)

vLLM ships compiled CUDA extensions (`vllm._C`) that are **Linux-only**, so the
`script_setup` pipeline cannot run on native Windows (`ModuleNotFoundError: No
module named 'vllm._C'`). The Docker image runs the exact same offline
`vllm.LLM` path inside a Linux GPU container, with the Hugging Face cache and
`data/` outputs mounted on the host for native-equivalent performance.

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

```powershell
.\scripts\docker-build.ps1
```

```bash
./scripts/docker-build.sh
```

The first build is large (CUDA + the full ML stack) and re-resolves
dependencies on Linux, so `uv.lock` (resolved on Windows) is intentionally not
used inside the image. After changing `pyproject.toml` pins (torch, vLLM, etc.),
rebuild the image so the container picks up the new stack.

### Run pipeline stages

The runner takes one flag per stage; combine them freely or use `--all`. Each
selected stage loads its own vLLM engine (`stage_1_vllm_config` /
`stage_2_vllm_config` in the YAML) and tears it down before the next stage
starts, and stages always run in ascending order.

```bash
cd docker

# Stage 1 only: idea generation -> data/<idea_id>/script.json
docker compose run --rm script-setup \
  python script_setup/script_setup_runner.py --config configs/script_setup_qwen3_4b.yaml --1

# Stage 2 only: titles from existing data/<idea_id>/script.json
docker compose run --rm script-setup \
  python script_setup/script_setup_runner.py --config configs/script_setup_qwen3_4b.yaml --2

# Stages 1 then 2 (ideas passed in memory, engine reinitialized between stages)
docker compose run --rm script-setup \
  python script_setup/script_setup_runner.py --config configs/script_setup_qwen3_4b.yaml --1 --2

# All implemented stages (this is the default `script-setup` command)
docker compose run --rm script-setup \
  python script_setup/script_setup_runner.py --config configs/script_setup_qwen3_4b.yaml --all
```

Stage 1 and stage 2 both read and write `data/<idea_id>/script.json` (stage 1 creates scripts with ideas only; stage 2 adds titles). Images and other artifacts live in the same `<idea_id>` folder.
Both are on the host via the bind mount. Running `docker compose run --rm
script-setup` with no command override executes `--all`.

### Offline inference and the model cache

Model weights are cached in the `hf_cache` Docker volume, so only the first run
downloads them. For gated models, copy `docker/.env.example` to `docker/.env`
and set `HF_TOKEN`. To prefetch weights and then run fully offline:

```bash
cd docker
docker compose run --rm prefetch-model      # downloads MODEL into hf_cache
# set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in docker/.env, then run
docker compose run --rm script-setup
```

### Other services

```bash
cd docker
docker compose run --rm shell               # interactive GPU shell
docker compose up jupyter                    # JupyterLab on http://localhost:8888
```

### Performance notes

- The compose services request all GPUs (`--gpus all`); set
  `CUDA_VISIBLE_DEVICES` in `docker/.env` to pin a specific GPU.
- `shm_size: "16gb"` is provided for vLLM; raise it for larger models.
- Keep the YAML knobs (`gpu_memory_utilization`, `enforce_eager`,
  `max_model_len`, `quantization`) tuned to your VRAM, e.g.
  [`src/configs/script_setup_qwen3_4b.yaml`](src/configs/script_setup_qwen3_4b.yaml).
- vLLM **0.8.5** loads Qwen3 with the native CUDA backend (no Transformers fallback).
- Compose sets `VLLM_USE_V1=0` by default. If you see `EngineCore failed to start` /
  `BackendCompilerFailed` during model load, confirm that variable is `0` and keep
  `enforce_eager: true` in the YAML until V1 + torch.compile is stable on your GPU.

## Virtual environments

| Venv | Use for |
|------|---------|
| `.venv` | SD 1.5, SDXL, ControlNet, LoRA, xFormers, **SVD**, transformers/BERT/GPT-style NLP, Coqui TTS (Bark + Fairseq models), JupyterLab |
| `.venv-stability` | Official **SV3D** and **SV4D** ([Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)) |

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
- **Video:** SVD via diffusers; I/O via `imageio`, `av`, `opencv-python`, `einops`, `kornia`
- **NLP:** `transformers`, `tokenizers`, `datasets`, `sentencepiece`
- **TTS:** `coqui-tts[languages]` (Bark + Fairseq TTS through Coqui; not legacy `TTS` on PyPI)
- **Utilities:** `jupyterlab`, `matplotlib`, `huggingface-hub`, `pillow`

Pinned PyTorch (default): **torch 2.6.0 + CUDA 12.4 + xformers 0.0.29.post2 + vLLM 0.8.5** (native Qwen3).

## Model weights

Weights are **not** downloaded by the install script. After install, use `huggingface-cli download` (hints printed at end of install) or your own `data/models/` layout.

## Project layout

```
pyproject.toml              # Main dependency manifest (uv)
requirements/
  constraints-main.txt      # Torch/xformers pins for install phase A
  stability-pt2.txt         # Stability generative-models deps (stability venv)
scripts/
  install.ps1
  install.sh
vendor/generative-models/   # Cloned by install (gitignored)
src/rnd4impact/             # Project package
```

## Optional extras

```powershell
uv sync --python .venv\Scripts\python.exe --extra sd-extras --extra dev
```
