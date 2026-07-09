"""Regression for configure_accelerate() when HF_HOME is set.

Issue #3502: HF_HOME destination was the cache directory itself, so
shutil.copyfile raised IsADirectoryError instead of writing
$HF_HOME/accelerate/default_config.yaml.
"""

import importlib.util
import logging
import os

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "setup_common",
    os.path.join(_repo_root, "setup", "setup_common.py"),
)
setup_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup_common)


def _clear_accelerate_env_candidates(monkeypatch):
    """Prefer HF_HOME by removing Windows-oriented path candidates."""
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)


def test_configure_accelerate_copies_to_hf_home_accelerate_yaml(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    _clear_accelerate_env_candidates(monkeypatch)

    setup_common.configure_accelerate(run_accelerate=False)

    dest = hf_home / "accelerate" / "default_config.yaml"
    assert dest.is_file()
    assert hf_home.is_dir()
    assert not hf_home.is_file()


def test_configure_accelerate_skips_copy_when_hf_home_config_exists(
    tmp_path, monkeypatch
):
    hf_home = tmp_path / "hf"
    dest = hf_home / "accelerate" / "default_config.yaml"
    dest.parent.mkdir(parents=True)
    dest.write_text("existing: true\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    _clear_accelerate_env_candidates(monkeypatch)

    setup_common.configure_accelerate(run_accelerate=False)

    assert dest.read_text(encoding="utf-8") == "existing: true\n"


def test_configure_accelerate_copy_failure_does_not_abort(
    tmp_path, monkeypatch, caplog
):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    _clear_accelerate_env_candidates(monkeypatch)

    def _boom(*_args, **_kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(setup_common.shutil, "copyfile", _boom)

    with caplog.at_level(logging.ERROR):
        setup_common.configure_accelerate(run_accelerate=False)

    dest = hf_home / "accelerate" / "default_config.yaml"
    assert not dest.exists()
    assert any(
        "accelerate" in record.getMessage().lower()
        or "copy" in record.getMessage().lower()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )
