# Musubi Tuner models and features

Features that require [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) rather than this GUI’s [sd-scripts](https://github.com/kohya-ss/sd-scripts) submodule are out of scope for kohya_ss until (and unless) they land in sd-scripts.

## Why this is out of scope

kohya_ss is a Gradio/CLI front end over the **sd-scripts** training surface. It does not vendor or wrap musubi-tuner. New model families that Kohya implements only in musubi-tuner (video, Qwen-Image DiT, Kontext, FramePack, Wan, etc.) need either:

1. Upstream support in sd-scripts, after which a GUI tab/controls can be added here, or
2. A separate GUI / workflow aimed at musubi-tuner.

Shipping half-wired GUI controls for a trainer this repo does not run would mislead users and create unmaintainable dual backends.

Examples of musubi-only (or musubi-first) requests:

| Concept | Notes |
|--------|--------|
| **Qwen-Image** (full DiT / LoRA / finetune) | sd-scripts uses the Qwen-Image **VAE** only as part of **Anima** training. Standalone Qwen-Image training is musubi-tuner (`docs/qwen_image.md`). Confirmed by Kohya on sd-scripts#2173 / #2182. |
| **FLUX.1 Kontext** | Training docs live under musubi-tuner (`docs/flux_kontext.md`), not the sd-scripts supported-model list. |
| **qinglong / qwen_shift timestep samplers** | Added in musubi-tuner advanced sampling; not part of sd-scripts FLUX/Anima timestep choices (`sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift`). |

## Prior requests

- #3421 — “Add supportion to f1 krea, qwen-image, etc.” (Qwen-Image portion; FLUX.1-Krea is already covered by the Flux path)
- #3317 — “Feature Request - Kontext Dev Lora training”
- #3387 — “Add the qinglong option to timestep sampling”
