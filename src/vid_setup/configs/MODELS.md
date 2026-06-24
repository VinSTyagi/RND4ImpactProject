# Video configs by VRAM tier

| Config | Model | VRAM | Notes |
|--------|-------|------|-------|
| `vid_setup_6gb.yaml` | SVD | ~6GB | Sequential CPU offload, 512×288, 14 frames |
| `vid_setup_12gb.yaml` | SVD | ~12GB | CPU offload, 1024×576, 14 frames |
| `vid_setup_24gb.yaml` | Wan 2.2 TI2V 5B | ~24GB | CPU offload, 832×480, 49 frames |
| `vid_setup_40gb.yaml` | Wan 2.2 TI2V 5B | ~40GB | CPU offload + VAE tiling, 1280×704, 81 frames |
| `vid_setup_80gb.yaml` | Wan 2.1 I2V 14B | ~80GB | Full GPU, 832×480, 49 frames |

Pick matching `script_setup`, `image_setup`, and `vid_setup` configs for your VRAM tier
(for example `configs/*_40gb.yaml` in each setup directory).
