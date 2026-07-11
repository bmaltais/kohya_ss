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

    # xformers dropdown ("none"/"sdpa"/"xformers") -> xformers/sdpa
    # mutually exclusive booleans (arch-matrix-leco.md #37-38).
    xformers_choice = values.get("xformers")
    out["xformers"] = True if xformers_choice == "xformers" else None
    out["sdpa"] = True if xformers_choice == "sdpa" else None

    # network_args / optimizer_args / lr_scheduler_args: space-separated,
    # quote-stripped textbox -> list of strings (arch-matrix-leco.md #15,
    # #21, #23). optimizer_args's old-GUI guard against `!= []` is
    # practically always true for a string widget (dead in practice, per
    # the matrix note) -- ported faithfully as a plain split, not a
    # conditional None.
    out["network_args"] = str(values.get("network_args") or "").replace('"', "").split()
    out["optimizer_args"] = (
        str(values.get("optimizer_args") or "").replace('"', "").split()
    )
    out["lr_scheduler_args"] = (
        str(values.get("lr_scheduler_args") or "").replace('"', "").split()
    )

    # wandb_run_name falls back to output_name when empty
    # (arch-matrix-leco.md #56).
    if not values.get("wandb_run_name"):
        out["wandb_run_name"] = values.get("output_name") or None

    return out
