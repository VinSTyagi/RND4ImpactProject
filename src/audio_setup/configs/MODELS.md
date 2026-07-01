# audio_setup VRAM tiers

| Config | VRAM | Notes |
|--------|------|-------|
| `audio_setup_6gb.yaml` | ~6GB | 1.7B VoiceDesign, float16, low memory caps |
| `audio_setup_12gb.yaml` | ~12GB | Default balanced tier |
| `audio_setup_24gb.yaml` | ~24GB | `max_num_seqs: 2` |
| `audio_setup_40gb.yaml` | ~40GB | `max_num_seqs: 4`, `enforce_eager: false` |
| `audio_setup_80gb.yaml` | ~80GB | `max_num_seqs: 8`, bfloat16 |

All tiers use `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`. Run via:

```bash
RND4IMPACT_VRAM_TIER=12gb ./scripts/run-full-pipeline.sh
```

Or audio only:

```bash
./scripts/run-full-pipeline.sh --only audio --tier 12gb
```
