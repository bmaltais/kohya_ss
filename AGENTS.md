# kohya_ss

## Purpose

Gradio-based GUI and CLI front end for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), used to configure and launch Stable Diffusion / SDXL / FLUX training runs (LoRA, DreamBooth, fine-tuning, TI) without hand-writing command lines.

## Ownership

- Maintainer: bmaltais
- `sd-scripts/` is a **git submodule** pointing at upstream kohya-ss/sd-scripts. Never edit files inside it — GUI-side changes only. If a fix requires a sd-scripts change, describe it as a snippet in the PR body instead of committing to the submodule.

## Local Contracts

- Python 3.10–3.11 (`pyproject.toml`), dependency management via `uv` (never bare `pip`).
- `kohya_gui.py` is the launcher entry point; it shares a name with the `kohya_gui/` package, so anything loading it dynamically (see `test/test_allowed_paths.py`) must import by file path, not `import kohya_gui`.
- Config precedence: `config.toml` (user, gitignored) vs `config example.toml` (tracked template) — copy, don't edit the example in place.
- Secrets (API tokens, HF keys) go through env vars / `config.toml`, never hardcoded.

## Work Guidance

- Black formatting on Python changes before commit.
- Keep GUI-only changes decoupled from submodule internals — call `sd-scripts` via its existing CLI/library surface, don't reach into its internals from `kohya_gui/`.
- Before a refactor touching >300 LOC, strip dead props/exports/debug logging first, as its own commit.

## Verification

- `.github/workflows/typos.yaml` runs `crate-ci/typos` on push/PR — fix flagged typos before merging.
- `tests/` holds the pytest regression suite; `test/test_allowed_paths.py` is a standalone unittest. Run both before shipping GUI changes (`pytest tests/ test/test_allowed_paths.py`).
- No `tsc`/`eslint` equivalent exists (pure Python project) — rely on the tests above plus manual GUI smoke-test via `gui.sh` / `gui.bat`.

## Child DOX Index

- `kohya_gui/AGENTS.md` — Gradio GUI package (tabs, shared widget classes, launcher helpers)
- `tools/AGENTS.md` — standalone CLI utility scripts (extraction, conversion, captioning)
- `docs/AGENTS.md` — feature guides and localized documentation
- `tests/AGENTS.md` — pytest regression suite
- `test/AGENTS.md` — manual end-to-end scratch fixtures and the allowed-paths unittest
- `sd-scripts/` — upstream submodule, out of scope for local AGENTS.md (do not create one; see Ownership above)
