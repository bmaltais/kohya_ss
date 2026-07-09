# tools/

## Purpose

Standalone CLI scripts for dataset prep, model/LoRA conversion, and one-off maintenance tasks. Independent of the Gradio app — each script is runnable directly with `python tools/<script>.py`.

## Ownership

Utility layer; not wired into the `kohya_gui/` tabs unless a `*_gui.py` file explicitly shells out to one (e.g. LoRA extraction/merge tools back the corresponding `extract_*_gui.py` / `merge_*_gui.py` tabs).

## Local Contracts

- Each script is self-contained with its own argparse CLI — no shared entry point or package `__init__.py`.
- `lycoris_utils.py` is a shared helper module imported by the `lycoris_locon_extract.py` / `merge_lycoris.py` scripts — keep LyCORIS-specific logic there rather than duplicating it per script.
- Image dataset scripts (`crop_images_to_n_buckets.py`, `group_images.py`, `group_images_recommended_size.py`, `create_txt_from_images.py`, `rename_depth_mask.py`) operate in place on a target folder passed via CLI arg — they mutate/rename files, so treat them as destructive unless run with a `--dry-run`-style flag where supported.
- Model/LoRA conversion and merge scripts (`extract_locon.py`, `extract_loha_from_model.py`, `extract_lora_from_models-new.py`, `extract_model_difference.py`, `merge_lycoris.py`, `resize_lora.py`, `prune.py`, `lcm_convert.py`) read/write `.safetensors` files and should not modify inputs in place — always write to a new output path.

## Work Guidance

- New one-off maintenance scripts belong here, not in `kohya_gui/`, unless they need a GUI tab.
- Prefer extending `lycoris_utils.py` for LyCORIS format changes rather than editing extraction logic per-script.

## Verification

None currently — no automated tests cover this folder; verify manually against a sample model/dataset before relying on a script's output.

## Child DOX Index

None — no subdirectories with independent scope.
