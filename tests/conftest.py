"""Shared pytest fixtures/helpers for kohya_gui tests.

`build_train_model_kwargs` fills a train_model()'s full positional-keyword
signature from a real preset JSON fixture in test/config/, patching any
param missing from the fixture (newer fields) with a type-appropriate
default so the call succeeds without a live Gradio session.
"""

import json
import inspect
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import toml

# Repo root (parent of tests/). Never hardcode machine-absolute paths in
# fixtures; resolve in-repo relatives from here so WSL/Linux/Windows agree.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Path-like kwargs that must exist on disk for train_model validation.
_REPO_PATH_KEYS = frozenset(
    {
        "dataset_config",
        "train_data_dir",
        "reg_data_dir",
        "logging_dir",
        "output_dir",
    }
)

# Heuristics for defaulting fields the fixture doesn't cover, or that the
# fixture stores as an empty string placeholder for a numeric field (Gradio
# coerces these at call time in the real app; direct calls must do it here).
BOOL_NAME_HINTS = (
    "checkbox",
    "cache_",
    "enable_",
    "use_",
    "save_state",
    "resume_from",
    "async_upload",
    "fused",
    "mem_eff_save",
    "apply_",
    "generate_",
    "disable_mmap",
    "split_qkv",
    "train_t5xxl",
    "train_double_block",
    "train_single_block",
)


def resolve_repo_path(value: str) -> str:
    """Map a fixture path to an absolute path under the repo when possible.

    Accepts portable relatives like ``./test/config/dataset.toml`` or
    ``test/config/dataset.toml``. Does not invent machine-specific roots.
    Absolute paths that already exist are left unchanged.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    raw = value.strip().replace("\\", "/")
    # Drop a single leading ./ used in some fixtures
    if raw.startswith("./"):
        raw = raw[2:]
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate)
    under_repo = (_REPO_ROOT / raw).resolve()
    if under_repo.exists():
        return str(under_repo)
    return value


def build_train_model_kwargs(
    train_model_fn,
    fixture_path: str,
    numeric_fixups: tuple = (),
    string_overrides: tuple = (),
    overrides: dict = None,
) -> dict:
    """Build a full kwargs dict for `train_model_fn` from a JSON fixture.

    - Fields present in the fixture are used as-is, unless listed in
      `numeric_fixups` and stored as "" (a blank Gradio Number field),
      in which case they're coerced to 0.
    - Fields missing from the fixture default to False (BOOL_NAME_HINTS),
      "" (`string_overrides`), or 0 otherwise.
    - In-repo path fields are resolved via :func:`resolve_repo_path`.
    - `overrides` wins over everything.
    """
    sig = inspect.signature(train_model_fn)
    params = [p for p in sig.parameters if p not in ("headless", "print_only")]

    with open(fixture_path, encoding="utf-8") as f:
        cfg = json.load(f)

    kwargs = {}
    for p in params:
        if p in cfg:
            v = cfg[p]
            if p in numeric_fixups and v == "":
                v = 0
            elif p in _REPO_PATH_KEYS and isinstance(v, str) and v:
                v = resolve_repo_path(v)
            kwargs[p] = v
        elif any(hint in p for hint in BOOL_NAME_HINTS):
            kwargs[p] = False
        elif p in string_overrides:
            kwargs[p] = ""
        else:
            kwargs[p] = 0

    kwargs["headless"] = True
    kwargs["print_only"] = True
    if overrides:
        kwargs.update(overrides)
        # Re-resolve path overrides so callers can pass relatives too
        for key in _REPO_PATH_KEYS:
            if key in overrides and isinstance(kwargs.get(key), str) and kwargs[key]:
                kwargs[key] = resolve_repo_path(kwargs[key])
    return kwargs


def mock_executor(gui_module) -> MagicMock:
    """Patch `<gui_module>.executor` so train_model()'s is_running() guard passes."""
    executor = MagicMock()
    executor.is_running.return_value = False
    gui_module.executor = executor
    return executor


def run_train_model_and_load_toml(gui_module, kwargs: dict) -> dict:
    """Call `gui_module.train_model(**kwargs)` with print_only=True in a
    scratch output_dir, then load and return the one TOML config it wrote.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        kwargs = dict(kwargs, output_dir=tmpdir, output_name="testout")
        mock_executor(gui_module)
        # train_model aborts if accelerate is missing from PATH; pin a fake
        # path so the suite does not depend on the host environment.
        with patch.object(gui_module, "get_executable_path", return_value="accelerate"):
            gui_module.train_model(**kwargs)
        toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
        assert len(toml_files) == 1, f"expected exactly one .toml, got {toml_files}"
        with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
            return toml.load(f)


def run_train_model_and_load_saved_json(gui_module, kwargs: dict) -> dict:
    """Call `gui_module.train_model(**kwargs)` with print_only=False in a
    scratch output_dir, then load and return the JSON training config
    `SaveConfigFile` wrote (the preset callers re-load via "Load config").
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        kwargs = dict(
            kwargs, output_dir=tmpdir, output_name="testout", print_only=False
        )
        mock_executor(gui_module)
        with patch.object(gui_module, "get_executable_path", return_value="accelerate"):
            gui_module.train_model(**kwargs)
        json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
        assert len(json_files) == 1, f"expected exactly one .json, got {json_files}"
        with open(os.path.join(tmpdir, json_files[0]), encoding="utf-8") as f:
            return json.load(f)
