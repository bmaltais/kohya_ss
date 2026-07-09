# kohya_gui/

## Purpose

Gradio UI package that builds every tab in the kohya_ss app and translates form inputs into `sd-scripts` CLI invocations. Owned by the root `AGENTS.md`; see it for submodule and dependency rules.

## Ownership

Part of the GUI front end. No files here belong to `sd-scripts` — this package only calls it (via `class_command_executor.py` / subprocess), never imports its internals directly.

## Local Contracts

- `<feature>_gui.py` files (e.g. `lora_gui.py`, `dreambooth_gui.py`, `finetune_gui.py`) each build one Gradio tab and are self-contained entry points.
- `class_*.py` files are shared, stateful widget groups reused across multiple tabs (e.g. `class_basic_training.py`, `class_advanced_training.py`, `class_source_model.py`, `class_folders.py`, `class_accelerate_launch.py`) — extend these instead of duplicating widget layout in a feature file.
- `common_gui.py` holds shared helpers (path pickers, config load/save, validation) used across both `class_*` and `*_gui.py` files — put new cross-tab utilities here, not in a feature file.
- `class_command_executor.py` is the only place that shells out to `sd-scripts` training scripts; new training features should route through it rather than spawning subprocesses ad hoc.
- `class_gui_config.py` / `class_configuration_file.py` own reading and writing the JSON/TOML preset files under `presets/` and `config_files/` — don't hand-roll config serialization elsewhere.
- `localization.py` / `localization_ext.py` own i18n string loading from `localizations/`.

## Work Guidance

- When adding a new training method tab, follow the existing `*_gui.py` + shared `class_*` composition pattern rather than inlining a monolithic tab.
- Keep widget classes decoupled from each other; compose them inside the feature file instead of having one `class_*` reach into another.

## Verification

- No dedicated unit tests for this package yet beyond `test/test_allowed_paths.py` (launcher-level) — smoke-test new/changed tabs by launching the GUI (`gui.sh` / `gui.bat`) and exercising the tab manually.
- **Launching as an agent (non-interactive/background shell):** use the same invocation as `gui-uv.bat`, not a bare `uv run python kohya_gui.py`:
  `uv run --link-mode=copy --index-strategy unsafe-best-match kohya_gui.py --noverify`
  Without `--noverify`, kohya_gui.py re-validates/reinstalls requirements every launch (slow, and reinstalls the editable `sd-scripts` package unnecessarily).
  When stdout is piped to a background-process log file (not a TTY), Python buffers it — the `* Running on local URL: ...` line may not appear in the log for a long time even though the server is already up. Don't treat a quiet log as a hang: poll the HTTP endpoint instead (`curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/`, expect `200`) rather than grepping the log for the ready banner.

## Child DOX Index

None — no subdirectories with independent scope.
