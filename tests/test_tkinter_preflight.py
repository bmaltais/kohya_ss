"""Tests for setup.sh Tkinter preflight (issue #3459).

Detection must use `import tkinter` on the installer interpreter, not dpkg/rpm
package presence. The uv launcher path must not run this gate at all.
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREFLIGHT = _REPO_ROOT / "setup" / "tkinter_preflight.sh"
_PREFLIGHT_REL = "setup/tkinter_preflight.sh"
_SETUP_SH = _REPO_ROOT / "setup.sh"
_GUI_UV_SH = _REPO_ROOT / "gui-uv.sh"


def _bash_available() -> bool:
    try:
        subprocess.run(
            ["bash", "-c", "echo ok"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return False


def _to_bash_path(path: Path | str) -> str:
    """Translate a host path into a form the local ``bash`` can open.

    On Windows, ``bash`` is often WSL or Git Bash; neither accepts raw
    ``D:\\...`` paths with backslashes. Prefer ``wslpath`` / ``cygpath``
    with forward-slash Windows paths, then fall back to ``/d/...``.
    """
    resolved = Path(path).resolve()
    if os.name != "nt":
        return str(resolved)

    # Forward slashes: required for reliable wslpath from Windows Python.
    win = str(resolved).replace("\\", "/")
    try:
        result = subprocess.run(
            ["bash", "-c", f"wslpath -u {shlex.quote(win)}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(_REPO_ROOT),
        )
        out = result.stdout.strip()
        # Reject "." / empty — seen when wslpath mishandles backslash paths.
        if out and out not in {".", "./"}:
            return out
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        pass
    try:
        result = subprocess.run(
            ["cygpath", "-u", win],
            capture_output=True,
            text=True,
            check=True,
        )
        out = result.stdout.strip()
        if out and out not in {".", "./"}:
            return out
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        pass
    # Fallback: D:/foo/bar -> /d/foo/bar (Git Bash) or leave as-is.
    if len(win) >= 2 and win[1] == ":":
        return f"/{win[0].lower()}{win[2:]}"
    return win


def _write_fake_python(directory: Path, *, import_ok: bool) -> Path:
    """Create a fake python executable that accepts `-c 'import tkinter'`."""
    path = directory / "fake-python"
    exit_code = 0 if import_ok else 1
    # Invoked as: fake-python -c "import tkinter"
    body = textwrap.dedent(f"""\
        #!/bin/sh
        # Fake interpreter for tkinter preflight unit tests.
        if [ "$1" = "-c" ]; then
          case "$2" in
            *tkinter*)
              exit {exit_code}
              ;;
          esac
        fi
        exit 1
        """)
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # WSL may ignore NTFS execute bits until chmod is run from bash.
    bash_path = _to_bash_path(path)
    subprocess.run(
        ["bash", "-c", f"chmod +x {shlex.quote(bash_path)}"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
    )
    return path


def _source_and_run(
    snippet: str, *, env: dict | None = None
) -> subprocess.CompletedProcess:
    # Source via repo-relative path + cwd=repo so Windows/WSL/Git-Bash agree
    # without embedding machine-absolute paths in the shell script text.
    script = f"source {_PREFLIGHT_REL}\n{snippet}\n"
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=run_env,
        check=False,
        cwd=str(_REPO_ROOT),
    )


@unittest.skipUnless(_bash_available(), "bash is required for tkinter preflight tests")
class TestPythonHasTkinter(unittest.TestCase):
    def test_preflight_script_exists(self):
        self.assertTrue(_PREFLIGHT.is_file(), f"missing {_PREFLIGHT}")

    def test_passes_when_interpreter_can_import_tkinter(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _write_fake_python(Path(tmp), import_ok=True)
            result = _source_and_run(f'python_has_tkinter "{_to_bash_path(fake)}"')
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
            )

    def test_fails_when_interpreter_cannot_import_tkinter(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _write_fake_python(Path(tmp), import_ok=False)
            result = _source_and_run(f'python_has_tkinter "{_to_bash_path(fake)}"')
            self.assertNotEqual(result.returncode, 0)

    def test_fails_for_missing_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no-such-python"
            result = _source_and_run(f'python_has_tkinter "{_to_bash_path(missing)}"')
            self.assertNotEqual(result.returncode, 0)

    def test_passes_on_real_python_when_tkinter_present(self):
        """Skip if host python lacks tkinter or bash cannot exec it.

        On Windows with WSL-as-bash, the host ``python.exe`` may not be
        executable from bash (exit 126). Fake-python tests cover the
        preflight contract in that case.
        """
        import sys

        probe = subprocess.run(
            [sys.executable, "-c", "import tkinter"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            self.skipTest(f"{sys.executable} cannot import tkinter")

        bash_py = _to_bash_path(sys.executable)
        can_run = subprocess.run(
            ["bash", "-c", f"{shlex.quote(bash_py)} -c 'import tkinter'"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(_REPO_ROOT),
        )
        if can_run.returncode != 0:
            self.skipTest(
                f"bash cannot exec host python at {bash_py!r} "
                f"(rc={can_run.returncode}); covered by fake-python tests"
            )

        result = _source_and_run(f'python_has_tkinter "{bash_py}"')
        self.assertEqual(result.returncode, 0)


@unittest.skipUnless(_bash_available(), "bash is required for tkinter preflight tests")
class TestRequirePythonTkinter(unittest.TestCase):
    def test_require_succeeds_without_dpkg_when_import_works(self):
        """Core #3459 contract: capability wins over package-manager state."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = _write_fake_python(Path(tmp), import_ok=True)
            result = _source_and_run(
                f'require_python_tkinter "{_to_bash_path(fake)}" "Ubuntu" "debian"',
                env={"RUNPOD": "false"},
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
            )
            self.assertIn("Python TK found", result.stdout)

    def test_require_fails_with_ubuntu_hint_when_import_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _write_fake_python(Path(tmp), import_ok=False)
            result = _source_and_run(
                f'require_python_tkinter "{_to_bash_path(fake)}" "Ubuntu" "debian"',
                env={"RUNPOD": "false"},
            )
            self.assertNotEqual(result.returncode, 0)
            combined = result.stdout + result.stderr
            self.assertIn("python3-tk", combined)
            self.assertIn("import tkinter", combined.lower().replace("`", ""))

    def test_require_fails_with_fedora_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _write_fake_python(Path(tmp), import_ok=False)
            result = _source_and_run(
                f'require_python_tkinter "{_to_bash_path(fake)}" "Fedora" "rhel"',
                env={"RUNPOD": "false"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("python3-tkinter", result.stdout + result.stderr)

    def test_hint_family_mapping(self):
        cases = [
            ("Ubuntu", "debian", "ubuntu"),
            ("fedora", "rhel", "fedora"),
            ("arch", "arch", "arch"),
            ("opensuse-tumbleweed", "suse", "opensuse"),
            ("UnknownOS", "None", "generic"),
        ]
        for distro, family, expected in cases:
            with self.subTest(distro=distro, family=family):
                result = _source_and_run(
                    f'tkinter_hint_family_from_distro "{distro}" "{family}"'
                )
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout.strip(), expected)


class TestSetupShWiresCapabilityCheck(unittest.TestCase):
    def test_setup_sh_sources_preflight_helper(self):
        text = _SETUP_SH.read_text(encoding="utf-8")
        self.assertIn("tkinter_preflight.sh", text)
        self.assertIn("require_python_tkinter", text)

    def test_setup_sh_does_not_gate_on_dpkg_query(self):
        text = _SETUP_SH.read_text(encoding="utf-8")
        # Package manager queries must not be the success condition.
        self.assertNotIn("dpkg-query", text)
        self.assertNotIn("rpm -qa | grep -qi python3-tkinter", text)
        self.assertNotIn("pacman -Qi tk", text)
        self.assertNotIn("rpm -qa | grep -qi python-tk", text)


class TestUvPathBypassesSetupTkinterGate(unittest.TestCase):
    """uv sync / gui-uv.sh must not inherit the setup.sh dpkg false-negative.

    gui-uv.sh never invokes setup.sh; it only runs ``uv run ... kohya_gui.py``.
    Missing system python3-tk therefore cannot block the uv install path.
    """

    def test_gui_uv_sh_does_not_invoke_setup_sh(self):
        text = _GUI_UV_SH.read_text(encoding="utf-8")
        self.assertNotIn("setup.sh", text)
        self.assertNotIn("dpkg-query", text)
        self.assertNotIn("python3-tk", text)
        self.assertIn("uv run", text)

    def test_gui_uv_sh_has_no_tkinter_package_preflight(self):
        text = _GUI_UV_SH.read_text(encoding="utf-8")
        self.assertNotIn("tkinter", text.lower())
        self.assertNotIn("require_python_tkinter", text)


if __name__ == "__main__":
    unittest.main()
