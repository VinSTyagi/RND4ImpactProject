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

Pinned PyTorch (default): **torch 2.5.1 + CUDA 12.4 + xformers 0.0.28.post3** — see [`requirements/constraints-main.txt`](requirements/constraints-main.txt).

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
