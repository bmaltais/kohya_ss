"""Pure derivation functions for Anima LLLite fields that are not a direct
1:1 FieldSpec -> TOML key mapping (per wargame plan Move 5's counter-move
for derived/composite fields; see wargame/reference/arch-matrix-anima_lllite.md
section 3 for the source documentation of each derivation).

`derive(values, arch_key)` takes the raw widget-value dict and returns a
dict of TOML-key -> value overrides to merge on top of the plain FieldSpec
values before calling `build_run_config`.

This tab has exactly one architecture ("anima_lllite"), so `arch_key` is
accepted for signature parity with the other five training types' derive()
functions but unused here.

0 -> None sentinels (max_train_epochs, max_train_steps, seed) are NOT
reimplemented here: build_run_config's default falsy-drop already treats
0 as falsy (`0 == False`), matching the old GUI's blanket
`["", False, None]` filter exactly -- this is the same "falsy-keep pitfall"
the wargame plan flags, ported faithfully rather than fixed.
"""


def derive(values: dict, arch_key: str) -> dict:
    out = {}

    # dataset_config vs train_data_dir/conditioning_data_dir: mutually
    # exclusive dataset-source choice modeled as three separate widgets
    # (arch-matrix-anima_lllite.md section 3).
    if values.get("dataset_config"):
        out["train_data_dir"] = None
        out["conditioning_data_dir"] = None

    # min_bucket_reso / max_bucket_reso: only meaningful when enable_bucket
    # is True; otherwise both collapse to None regardless of widget value.
    if not values.get("enable_bucket"):
        out["min_bucket_reso"] = None
        out["max_bucket_reso"] = None

    # attn_mode: the literal default "torch" is translated to None on
    # write so the trainer's own default applies -- a value-based special
    # case, not an emptiness check (arch-matrix-anima_lllite.md #74).
    if values.get("attn_mode") == "torch":
        out["attn_mode"] = None

    # compile_backend/compile_mode/compile_dynamic/compile_fullgraph/
    # compile_cache_size_limit: all gated on the parent `compile` checkbox;
    # compile_dynamic is further suppressed at "auto", compile_cache_size_limit
    # further suppressed when falsy.
    if not values.get("compile"):
        out["compile_backend"] = None
        out["compile_mode"] = None
        out["compile_dynamic"] = None
        out["compile_fullgraph"] = None
        out["compile_cache_size_limit"] = None
    elif values.get("compile_dynamic") == "auto":
        out["compile_dynamic"] = None

    # show_timesteps_resolution depends on the sibling show_timesteps
    # field, not just its own value.
    if not values.get("show_timesteps"):
        out["show_timesteps_resolution"] = None

    # optimizer_args / lr_scheduler_args: space-separated, quote-stripped
    # textbox -> list of strings, explicitly None when unset -- unlike
    # LeCo/LoRA/DreamBooth/Finetune, anima_lllite_gui.py guards BOTH fields
    # with an explicit `not in ("", [], None) else None` check
    # (anima_lllite_gui.py:679-688), not just optimizer_args.
    for name in ("optimizer_args", "lr_scheduler_args"):
        raw = values.get(name)
        if raw in ("", [], None):
            out[name] = None
        else:
            out[name] = str(raw).replace('"', "").split()

    # wandb_run_name falls back to output_name when empty.
    if not values.get("wandb_run_name"):
        out["wandb_run_name"] = values.get("output_name") or None

    return out
