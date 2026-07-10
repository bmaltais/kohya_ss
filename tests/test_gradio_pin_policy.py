"""Regression for issue #3460: Gradio pins must stop pip multi-version backtracking.

Open lower-only ranges (gradio>=X) force pip to try many candidates against
exact pins on shared deps (huggingface-hub, aiofiles, …). Exact pins for
gradio and gradio-client keep classic setup.sh resolution deterministic and
aligned with uv.lock.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None  # type: ignore[assignment]

_repo_root = Path(__file__).resolve().parents[1]

# Align with uv.lock resolved versions (gradio 5.34.2 / client 1.10.3)
_GRADIO_EXPECTED = Version("5.34.2")
_GRADIO_CLIENT_EXPECTED = Version("1.10.3")


def _requirements_text() -> str:
    return (_repo_root / "requirements.txt").read_text(encoding="utf-8")


def _pyproject_deps() -> list[str]:
    """Return project.dependencies entries (tomllib on 3.11+, regex on 3.10)."""
    text = (_repo_root / "pyproject.toml").read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(text)
        return list(data["project"]["dependencies"])
    match = re.search(r"(?m)^dependencies\s*=\s*\[(.*?)^\]", text, re.DOTALL)
    if not match:
        raise AssertionError("dependencies array not found in pyproject.toml")
    return re.findall(r'"([^"]+)"', match.group(1))


def _parse_requirements_pin(package: str) -> Version:
    """Parse an exact pin (pkg==X.Y.Z) from requirements.txt."""
    pattern = re.compile(
        rf"^{re.escape(package)}\s*==\s*([0-9][^\s#]+)\s*(?:#.*)?$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(_requirements_text())
    if not match:
        raise AssertionError(
            f"{package} exact pin (pkg==version) not found in requirements.txt"
        )
    return Version(match.group(1))


def _parse_pyproject_pin(package: str) -> Version:
    """Parse an exact pin (pkg==X.Y.Z) from pyproject.toml dependencies."""
    package_norm = package.lower().replace("_", "-")
    for dep in _pyproject_deps():
        dep_name = re.split(r"[<>=!\[]", dep, maxsplit=1)[0].strip()
        if dep_name.lower().replace("_", "-") != package_norm:
            continue
        match = re.search(r"==\s*([0-9][^\s;]+)", dep)
        if not match:
            raise AssertionError(
                f"{package} is declared in pyproject.toml but is not an exact == pin: {dep!r}"
            )
        return Version(match.group(1))
    raise AssertionError(f"{package} not found in pyproject.toml dependencies")


def _uv_lock_package_version(package: str) -> Version:
    """Read version for a top-level package block in uv.lock."""
    text = (_repo_root / "uv.lock").read_text(encoding="utf-8")
    # uv.lock uses TOML tables: name = "pkg" then version = "X.Y.Z"
    pattern = re.compile(
        rf'(?m)^name = "{re.escape(package)}"\nversion = "([^"]+)"',
    )
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"{package} not found as a package entry in uv.lock")
    return Version(match.group(1))


class TestGradioPinPolicy(unittest.TestCase):
    def test_requirements_txt_pins_gradio_exactly(self) -> None:
        pin = _parse_requirements_pin("gradio")
        self.assertEqual(
            pin,
            _GRADIO_EXPECTED,
            f"requirements.txt gradio=={pin} must be =={_GRADIO_EXPECTED} "
            f"(issue #3460: open >= ranges cause pip backtracking)",
        )

    def test_requirements_txt_pins_gradio_client_exactly(self) -> None:
        pin = _parse_requirements_pin("gradio-client")
        self.assertEqual(
            pin,
            _GRADIO_CLIENT_EXPECTED,
            f"requirements.txt gradio-client=={pin} must be =={_GRADIO_CLIENT_EXPECTED}",
        )

    def test_pyproject_pins_gradio_exactly(self) -> None:
        pin = _parse_pyproject_pin("gradio")
        self.assertEqual(pin, _GRADIO_EXPECTED)

    def test_pyproject_pins_gradio_client_exactly(self) -> None:
        pin = _parse_pyproject_pin("gradio-client")
        self.assertEqual(pin, _GRADIO_CLIENT_EXPECTED)

    def test_gradio_pin_matches_between_requirements_and_pyproject(self) -> None:
        self.assertEqual(
            _parse_requirements_pin("gradio"),
            _parse_pyproject_pin("gradio"),
            "gradio pin must match between requirements.txt and pyproject.toml",
        )

    def test_gradio_client_pin_matches_between_requirements_and_pyproject(
        self,
    ) -> None:
        self.assertEqual(
            _parse_requirements_pin("gradio-client"),
            _parse_pyproject_pin("gradio-client"),
            "gradio-client pin must match between requirements.txt and pyproject.toml",
        )

    def test_uv_lock_gradio_matches_declared_pin(self) -> None:
        locked = _uv_lock_package_version("gradio")
        declared = _parse_requirements_pin("gradio")
        self.assertEqual(
            locked,
            declared,
            f"uv.lock gradio=={locked} must match requirements pin =={declared}",
        )

    def test_uv_lock_gradio_client_matches_declared_pin(self) -> None:
        locked = _uv_lock_package_version("gradio-client")
        declared = _parse_requirements_pin("gradio-client")
        self.assertEqual(
            locked,
            declared,
            f"uv.lock gradio-client=={locked} must match requirements pin =={declared}",
        )

    def test_linux_setup_uses_bulk_requirements_install(self) -> None:
        """Guard: setup must stay one bulk pip -r, not per-package gradio loops."""
        src = (_repo_root / "setup" / "setup_linux.py").read_text(encoding="utf-8")
        self.assertIn("install_requirements_inbulk", src)
        self.assertNotIn(
            'install("gradio"',
            src,
            "setup_linux must not install gradio as a separate package step",
        )


if __name__ == "__main__":
    unittest.main()
