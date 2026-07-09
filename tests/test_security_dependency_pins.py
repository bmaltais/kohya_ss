"""Regression for issue #3479: dependency pins must meet known CVE floors.

Shared pins live in requirements.txt (classic setup path) and pyproject.toml
(uv/Docker path). Both must stay in lockstep and above security floors.

Residual risk (not cleared by pin floors):
- CVE-2026-0994 (protobuf JSON recursion) — no upstream patch at triage time
- CVE-2024-8020 (pytorch-lightning DoS) — mitigated by not installing Lightning
  (local stub only under sd-scripts/pytorch_lightning/)
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

# CVE floors from triage of bmaltais/kohya_ss#3479
_ONNX_MIN = Version("1.17.0")  # CVE-2024-7776, CVE-2024-5187
_PROTOBUF_MIN = Version("4.25.8")  # CVE-2025-4565
_TRANSFORMERS_MIN = Version("4.53.0")  # transformers ReDoS / deserial CVEs
_SENTENCEPIECE_MIN = Version("0.2.1")  # CVE-2026-1260


def _requirements_text() -> str:
    return (_repo_root / "requirements.txt").read_text(encoding="utf-8")


def _pyproject_deps() -> list[str]:
    """Return project.dependencies entries (tomllib on 3.11+, regex on 3.10)."""
    text = (_repo_root / "pyproject.toml").read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(text)
        return list(data["project"]["dependencies"])
    # Python 3.10: no stdlib tomllib; extract the dependencies = [ ... ] block.
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
    # Normalize underscores/hyphens for matching package names
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


def _declared_dependency_names() -> set[str]:
    """Normalized package names declared in both install sources."""
    names: set[str] = set()
    for line in _requirements_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        name = re.split(r"[<>=!\[]", stripped, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower().replace("_", "-"))
    for dep in _pyproject_deps():
        name = re.split(r"[<>=!\[]", dep, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower().replace("_", "-"))
    return names


class TestSecurityDependencyPins(unittest.TestCase):
    def test_onnx_pin_meets_security_floor(self) -> None:
        req = _parse_requirements_pin("onnx")
        proj = _parse_pyproject_pin("onnx")
        self.assertGreaterEqual(
            req,
            _ONNX_MIN,
            f"requirements.txt onnx=={req} is below {_ONNX_MIN} (CVE-2024-7776)",
        )
        self.assertGreaterEqual(
            proj,
            _ONNX_MIN,
            f"pyproject.toml onnx=={proj} is below {_ONNX_MIN} (CVE-2024-7776)",
        )

    def test_onnx_pin_matches_between_requirements_and_pyproject(self) -> None:
        self.assertEqual(
            _parse_requirements_pin("onnx"),
            _parse_pyproject_pin("onnx"),
            "onnx pin must match between requirements.txt and pyproject.toml",
        )

    def test_protobuf_pin_meets_security_floor(self) -> None:
        req = _parse_requirements_pin("protobuf")
        proj = _parse_pyproject_pin("protobuf")
        self.assertGreaterEqual(
            req,
            _PROTOBUF_MIN,
            f"requirements.txt protobuf=={req} is below {_PROTOBUF_MIN} (CVE-2025-4565)",
        )
        self.assertGreaterEqual(
            proj,
            _PROTOBUF_MIN,
            f"pyproject.toml protobuf=={proj} is below {_PROTOBUF_MIN} (CVE-2025-4565)",
        )

    def test_protobuf_pin_matches_between_requirements_and_pyproject(self) -> None:
        self.assertEqual(
            _parse_requirements_pin("protobuf"),
            _parse_pyproject_pin("protobuf"),
            "protobuf pin must match between requirements.txt and pyproject.toml",
        )

    def test_transformers_pin_meets_security_floor(self) -> None:
        req = _parse_requirements_pin("transformers")
        proj = _parse_pyproject_pin("transformers")
        self.assertGreaterEqual(req, _TRANSFORMERS_MIN)
        self.assertGreaterEqual(proj, _TRANSFORMERS_MIN)
        self.assertEqual(req, proj)

    def test_sentencepiece_pin_meets_security_floor(self) -> None:
        req = _parse_requirements_pin("sentencepiece")
        proj = _parse_pyproject_pin("sentencepiece")
        self.assertGreaterEqual(req, _SENTENCEPIECE_MIN)
        self.assertGreaterEqual(proj, _SENTENCEPIECE_MIN)
        self.assertEqual(req, proj)

    def test_pytorch_lightning_not_declared_as_dependency(self) -> None:
        names = _declared_dependency_names()
        self.assertNotIn(
            "pytorch-lightning",
            names,
            "pytorch-lightning must not be reintroduced (CVE-2024-8019 / CVE-2024-8020)",
        )


if __name__ == "__main__":
    unittest.main()
