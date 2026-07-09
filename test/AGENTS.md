# test/

## Purpose

Manual end-to-end scratch space for exercising real training runs through the GUI/CLI, plus one standalone unittest for launcher behavior. Distinct from `tests/` (plural) at the repo root, which is the automated pytest suite — see `tests/AGENTS.md`.

## Ownership

Mixed: `test_allowed_paths.py` is a real regression test; everything else (`config/`, `ft/`, `log/`, `logs/`, `masked_loss/`, `output/`, `img*/`) is generated/scratch data from manual runs, not curated fixtures.

## Local Contracts

- `test_allowed_paths.py` loads the root `kohya_gui.py` launcher by file path (via `importlib.util.spec_from_file_location`), not `import kohya_gui`, because the launcher script and the `kohya_gui/` package share a name — follow the same pattern for any new launcher-level test.
- `config/` holds sample training preset JSON/TOML files (dreambooth, LoRA, LoCon, LoHa, LoKR, DyLoRA, TI, SDXL variants) used as real inputs when manually exercising a tab end-to-end.
- `masked_loss/` holds a small fixed sample image set used to manually verify masked-loss training.
- `logs/`, `log/`, `ft/`, `output/` are run artifacts (tensorboard logs, checkpoints, generated configs) from prior manual test runs — timestamped, disposable, safe to prune; not meant to be committed as fixtures going forward.
- `img with spaces/` exists specifically to catch path-quoting bugs in generated CLI commands — keep the space in the folder name.

## Work Guidance

- New launcher/CLI-argument-forwarding tests belong here as standalone unittests, following `test_allowed_paths.py`'s import pattern.
- New pure-logic unit tests for `kohya_gui/` modules belong in `tests/` instead.
- Don't commit new large run artifacts under `logs/`/`output/`/`ft/` — clean up scratch runs rather than checking them in.

## Verification

- Run with `pytest test/test_allowed_paths.py` (or `python -m unittest test.test_allowed_paths`).

## Child DOX Index

None.
