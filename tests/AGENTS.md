# tests/

## Purpose

Pytest/unittest regression suite for `kohya_gui/` behavior that's cheap to isolate (no GPU, no model weights).

## Ownership

Mirrors `kohya_gui/` — tests here target that package's pure-Python logic (e.g. `class_tensorboard.py`'s AVX/tensorboard-availability detection).

## Local Contracts

- One `test_<module_under_test>.py` per targeted `kohya_gui/` module; use `importlib.reload` on the target module inside each test when the module has import-time side effects that need re-triggering under different mocked conditions (see `test_tensorboard_visibility.py`).
- Mock external/environment dependencies (`shutil.which`, `cpuinfo.get_cpu_info`, hardware/OS checks) — this suite must run without a GPU or real tensorboard/cpuinfo state.
- Distinct from `test/` (singular) at the repo root, which holds manual end-to-end scratch fixtures and outputs, not unit tests — see `test/AGENTS.md`.

## Work Guidance

- New pure-logic additions to `kohya_gui/` (config parsing, capability detection, validation helpers) should get a unit test here rather than only being smoke-tested through the GUI.

## Verification

- Run with `pytest tests/`.

## Child DOX Index

None.
