"""Pure derivation functions for LeCo fields that are not a direct 1:1
FieldSpec -> TOML key mapping (per wargame plan Move 5's counter-move for
derived/composite fields; see wargame/reference/arch-matrix-leco.md section
4 for the source documentation of each derivation).

`derive(values, arch_key)` takes the raw widget-value dict and returns a
dict of TOML-key -> value overrides to merge on top of the plain FieldSpec
values before calling `build_run_config`.

Architecture keys: sd15/sd2 (train_leco.py), sdxl (sdxl_train_leco.py) --
see leco_fields.py's ARCHITECTURE_CHOICES.

Type coercions (int()/float() casts on network_dim, network_alpha,
max_denoising_steps, etc. -- arch-matrix-leco.md section 4's last row) are
NOT reimplemented here: the generator already attaches to_toml/from_toml
coercers to those FieldSpecs from real argparse type introspection.
"""


def derive(values: dict, arch_key: str) -> dict:
    out = {}

    # v2 / v_parameterization: forced None under SDXL regardless of
    # checkbox state -- belt-and-suspenders with the old GUI's UI-side
    # toggle_v_family lock (arch-matrix-leco.md #2-3, section 4).
    if arch_key == "sdxl":
        out["v2"] = None
        out["v_parameterization"] = None

    # clip_skip: dropped both when 0 AND whenever SDXL is selected (SDXL
    # has no clip-skip concept) -- arch-matrix-leco.md #40.
    if arch_key == "sdxl":
        out["clip_skip"] = None

    # xformers/sdpa: FieldSpecs are now the direct trainer-facing booleans
    # (two independent checkboxes), not a dropdown -- the one-time
    # string->boolean split from a legacy JSON's "xformers" dropdown value
    # (arch-matrix-leco.md #37-38) happens in legacy_import.import_json,
    # against the raw JSON dict, not here. This function also runs on every
    # do_train/do_save via tab_builder.py's _build_values using the live
    # (already-boolean) checkbox values, so re-deriving from a
    # "xformers"/"sdpa" string match here would always fail and null both
    # fields back out.

    # network_args / optimizer_args / lr_scheduler_args: left as the raw
    # widget string here. FieldSpec.to_toml/from_toml (_to_arg_list/
    # _from_arg_list) own the textbox<->list round-trip for both the
    # run-config build and the import_json/normalize_widget_value display
    # path; duplicating the split here double-applied it (a widget already
    # showing "[]" from a prior derive() pass would get re-split into
    # ["[]"], crashing optimizer.py's `key, value = arg.split("=")` -- see
    # dreambooth optimizer_args regression, 2026-07-12).

    # wandb_run_name falls back to output_name when empty
    # (arch-matrix-leco.md #56).
    if not values.get("wandb_run_name"):
        out["wandb_run_name"] = values.get("output_name") or None

    return out
