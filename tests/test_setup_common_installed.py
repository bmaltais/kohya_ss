"""Regression for setup_common.installed() package discovery.

Covers the importlib.metadata migration (PR #3484): presence checks,
version constraints, and name-normalization fallbacks.
"""

import importlib.metadata
import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "setup_common",
    os.path.join(_repo_root, "setup", "setup_common.py"),
)
setup_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup_common)


def _fake_dist(version: str) -> MagicMock:
    dist = MagicMock()
    dist.version = version
    return dist


class TestInstalled(unittest.TestCase):
    def test_missing_package_returns_false(self):
        with patch.object(
            setup_common.importlib.metadata,
            "distribution",
            side_effect=importlib.metadata.PackageNotFoundError("nope"),
        ):
            self.assertFalse(setup_common.installed("not-a-real-package-xyz"))

    def test_installed_without_version_pin_returns_true(self):
        with patch.object(
            setup_common.importlib.metadata,
            "distribution",
            return_value=_fake_dist("1.2.3"),
        ):
            self.assertTrue(setup_common.installed("somepkg"))

    def test_exact_version_match(self):
        with patch.object(
            setup_common.importlib.metadata,
            "distribution",
            return_value=_fake_dist("1.2.3"),
        ):
            self.assertTrue(setup_common.installed("somepkg==1.2.3"))

    def test_exact_version_mismatch(self):
        with patch.object(
            setup_common.importlib.metadata,
            "distribution",
            return_value=_fake_dist("1.2.3"),
        ):
            self.assertFalse(setup_common.installed("somepkg==9.9.9"))

    def test_minimum_version_satisfied(self):
        with patch.object(
            setup_common.importlib.metadata,
            "distribution",
            return_value=_fake_dist("2.0.0"),
        ):
            self.assertTrue(setup_common.installed("somepkg>=1.5.0"))

    def test_minimum_version_not_satisfied(self):
        with patch.object(
            setup_common.importlib.metadata,
            "distribution",
            return_value=_fake_dist("1.0.0"),
        ):
            self.assertFalse(setup_common.installed("somepkg>=2.0.0"))

    def test_extras_brackets_are_stripped(self):
        with patch.object(
            setup_common.importlib.metadata,
            "distribution",
            return_value=_fake_dist("0.32.2"),
        ) as dist_mock:
            self.assertTrue(setup_common.installed("diffusers[torch]==0.32.2"))
            dist_mock.assert_called()
            self.assertEqual(dist_mock.call_args_list[0].args[0], "diffusers")

    def test_found_via_lowercase_fallback(self):
        def lookup(name):
            if name == "SomePkg":
                raise importlib.metadata.PackageNotFoundError(name)
            if name == "somepkg":
                return _fake_dist("1.0.0")
            raise importlib.metadata.PackageNotFoundError(name)

        with patch.object(
            setup_common.importlib.metadata, "distribution", side_effect=lookup
        ):
            self.assertTrue(setup_common.installed("SomePkg"))

    def test_found_via_underscore_to_dash_fallback(self):
        def lookup(name):
            if name in ("foo_bar", "foo_bar".lower()):
                raise importlib.metadata.PackageNotFoundError(name)
            if name == "foo-bar":
                return _fake_dist("3.1.0")
            raise importlib.metadata.PackageNotFoundError(name)

        with patch.object(
            setup_common.importlib.metadata, "distribution", side_effect=lookup
        ):
            self.assertTrue(setup_common.installed("foo_bar==3.1.0"))

    def test_real_stdlib_adjacent_package_smoke(self):
        # importlib.metadata itself is always importable; pip is commonly present
        # in project envs. Assert the helper returns a bool without raising.
        result = setup_common.installed("pip")
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
