# kohya_gui_v2 (preview)

`kohya_gui_v2` is a parallel, in-development GUI that re-implements the six
training tabs (LoRA, DreamBooth, Finetune, Textual Inversion, LeCo, Anima
LLLite) on a declarative field-registry pattern instead of six independently
hand-written, copy-pasted modules. It ships alongside the existing
`kohya_gui.py` / `kohya_gui/` GUI, which remains untouched and fully
supported -- v2 is currently a preview, not a replacement.

See `wargame/2026-07-11-musubi-style-gui-v2.md` for the full design
rationale and execution plan this was built against.

## Launching

```
uv run python kohya_gui_v2.py --headless
```

Same CLI flags as the existing `kohya_gui.py` (`--config`, `--listen`,
`--server_port`, `--inbrowser`, `--share`, `--headless`). All six training
tabs render, plus a "Preview" placeholder tab.

## The TOML round-trip contract

Every v2 tab's Save button writes a **flat TOML file** whose keys equal the
sd-scripts trainer's own `--config_file` argument names. That is: **the
file the GUI saves is the exact file the trainer consumes.** There is no
separate "run config" translation step at save time the way the legacy
JSON presets require -- Open, edit, Save, and Train all operate on the same
file format sd-scripts itself reads.

- **Save** writes every non-default field value, sorted by key, to a
  `.toml` file.
- **Open** accepts either a flat or sectioned TOML (or a legacy `.json`
  preset -- see below), filling every field from the file with a fallback
  to that field's default for anything absent.
- **Train** builds a filtered "run config" from the current widget values
  (drops falsy/default values, applies the field's own derivation logic --
  e.g. resolving `max_train_steps` from the dataset folder when left at
  `0`) and writes it to `<output_dir>/config_<type>_v2-<timestamp>.toml`,
  which is what's actually passed to `accelerate launch ... --config_file`.

## Legacy JSON import

The six existing `kohya_gui/*_gui.py` tabs persist presets as JSON in
their own GUI-parameter space (not the trainer's TOML key space -- e.g. the
widget is named `optimizer`, but the trainer's TOML key is
`optimizer_type`). Every v2 tab's Open dialog accepts these legacy `.json`
files directly, alongside `.toml`:

- Opening a `.json` preset from a v2 tab imports it through
  `kohya_gui_v2/legacy_import.py`, which maps GUI-parameter names onto the
  correct TOML-key FieldSpec names for that tab's training type and
  resolves the correct architecture (SDXL/SD3/FLUX.1/etc.) from the
  preset's own checkbox values -- the same detection logic validated by the
  Move 7 equivalence harnesses (`tests/test_v2_equivalence_*.py`).
- Any key in the JSON that the importer doesn't recognize (an older GUI
  version's field, a hand-edited file, or a genuine gap in the field
  registry) is **not** silently dropped -- it's surfaced as a GUI warning
  ("Ignored legacy keys not recognized by v2: ...") so the user knows their
  preset wasn't imported byte-perfect.
- Saving after opening a `.json` always writes a `.toml` -- v2 never writes
  JSON. The original `.json` preset is untouched.

`tools/convert_presets_to_toml.py` batch-converts every file under
`presets/**/*.json` to a sibling `.toml`, using the exact same
`legacy_import.import_json()` + `config_io.save_config()` code path the
Open dialog uses (there is deliberately no separate mapping table in the
converter). Run `uv run python tools/convert_presets_to_toml.py --dry-run`
to preview what would be converted, or without `--dry-run` to write the
`.toml` files. The original `.json` presets are never deleted.

## Known-defect register

The Move 7 equivalence harnesses (old GUI vs. v2, run over the `presets/`
corpus plus hand-built fixtures for LeCo/Textual Inversion/Anima LLLite)
found the following places where v2 *intentionally* differs from the
current `kohya_gui/*_gui.py` behavior, because the old GUI is provably
wrong there. Each entry cites the old-GUI location and the parser evidence
that confirms it:

| Training type | Key(s) | Old-GUI behavior | v2 behavior | Evidence |
|---|---|---|---|---|
| LeCo | `prompts_file` | Never passed on the CLI, only written to the TOML. Both `train_leco.py` and `sdxl_train_leco.py` declare `--prompts_file` as `required=True`, and argparse enforces this during its *first* `parse_args()` call -- before `--config_file`'s TOML is ever read. Every LeCo run launched from the old GUI currently fails with `SystemExit(2)`. | `leco_tab.py`'s `do_train` always emits `--prompts_file <value>` directly on the accelerate-launch command line, in addition to the TOML key. | `sd-scripts/train_leco.py:59`, `sd-scripts/sdxl_train_leco.py:61` (both `required=True`); `sd-scripts/library/args.py:1122-1186` (`read_config_from_file` only runs after the first `parse_args()` call). |
| LoRA | `lr_warmup_steps` | `lora_gui.py`'s non-`dataset_config` branch computes `lr_warmup_steps = lr_warmup / 100`, omitting `* max_train_steps` -- producing a near-zero warmup instead of a percentage of total training steps. | `lora_derivations.py` implements the correct formula (`lr_warmup / 100 * max_train_steps`). | `kohya_gui/lora_gui.py` ~line 1552. |
| DreamBooth | `split_mode`, `train_blocks` | GUI widgets exist and round-trip through JSON save/open, but their value is hardcoded to `None` before reaching `config_toml_data` -- `flux_train.py` (this tab's only FLUX.1 target) doesn't accept them; they're LoRA-only (`flux_train_network.py`) args. | v2 never emits these for DreamBooth (`dreambooth_fields.py`/`dreambooth_derivations.py` don't generate FieldSpecs for them). | `kohya_gui/dreambooth_gui.py` (comment citing `flux_train_network.py`-only status); confirmed via `wargame/reference/arch-matrix-dreambooth.md` lines 166-167, 200. |
| DreamBooth | `prior_loss_weight`, `stop_text_encoder_training`, `no_token_padding`, `learning_rate_te` (SDXL/SD3/FLUX.1 only) | Old GUI writes these universally, but SDXL/SD3/FLUX.1 reuse Finetune's entry scripts (`sdxl_train.py`/`sd3_train.py`/`flux_train.py`), which don't declare these DreamBooth-specific args at all. | v2's FieldSpecs for these keys are scoped to `sd15`/`sd2` only (confirmed via direct `setup_parser()` introspection, not just the gap-analysis doc). | `wargame/reference/gap-analysis-dreambooth.md` summary; `sd3_train.py`'s `setup_parser()` (checked directly for `learning_rate_te`). |
| Textual Inversion | `epoch`, `noise_offset_type` | Written universally, but neither `train_textual_inversion.py` nor `sdxl_train_textual_inversion.py` declare these args -- dead keys tolerated only because `--config_file` ignores unknown namespace attributes. | v2 omits both from every architecture's FieldSpecs. | `wargame/reference/gap-analysis-textual_inversion.md`. |
| Textual Inversion | `stop_text_encoder_training_pct` | Present in `train_model`'s signature and the JSON save/open round-trip (via a param-name alias), but explicitly commented out of `config_toml_data` -- neither trainer script accepts it. | Modeled as `gui_only=True` in `textual_inversion_fields.py`: preserved in the widget/JSON space, never projected into the run TOML. | `kohya_gui/textual_inversion_gui.py` (inline comment + `_TRAIN_TO_CONFIG_ALIASES`). |
| Finetune | `noise_offset_type` | Written universally; no finetune-family parser (`sdxl_train.py`/`sd3_train.py`/`flux_train.py`/`anima_train.py`/`lumina_train.py`/`fine_tune.py`) declares this arg. | v2 omits it from every architecture's FieldSpecs. | `wargame/reference/gap-analysis-finetune.md` summary. |

## `defer`-tagged gap args (maintainer follow-up)

The Move 4 gap analysis flagged one recurring `defer` item across three
training types -- `train_inpainting` as a *gap-candidate* (i.e. exposing it
for architectures where the current GUI never offers it at all, distinct
from the DreamBooth/Finetune tabs where it's already a real, user-facing
field):

| Training type | Architectures where `train_inpainting` is gap-candidate-deferred |
|---|---|
| LoRA | sdxl, flux1, sd3, hunyuan_image, lumina (5 of 8 architectures) |
| Textual Inversion | sd_v1v2, sdxl (both architectures) |
| Anima LLLite | anima_lllite (its one architecture) |

In each case the old GUI never exposes `train_inpainting` as a widget for
that architecture at all (LoRA's `network_module`-based architectures don't
have an inpainting toggle; neither does Textual Inversion or Anima LLLite),
but the underlying parser accepts it. Whether inpainting-mode training is
actually meaningful/tested for these architecture + training-type
combinations is a product question, not a code-archaeology one -- flagged
here for the maintainer to decide rather than silently exposing a widget
for an untested code path. DreamBooth and Finetune are unaffected: both
already have a real, user-facing `train_inpainting` checkbox today, so it's
a `port`/derivation case there, not a `defer`.

No other gap-candidate arg, across any of the six training types, needed a
`defer` disposition -- everything else was mechanically classified
`expose`, `expose-advanced`, or `exclude`.

## Scope and status

This is a preview. Not yet built: switchover/deprecation of the existing
GUI (explicitly out of scope -- a maintainer decision), a musubi-tuner
backend (the `BackendSpec` seam supports this as a future phase without
touching `fields.py`/`builder.py`/`config_io.py`, but it is not implemented
here), and end-to-end real-training validation beyond the equivalence
harnesses' `print_only=True` command construction checks.
