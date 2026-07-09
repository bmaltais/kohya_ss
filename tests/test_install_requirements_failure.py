"""Regression for issue #3499: setup must not treat pip failures as success."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_repo_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "setup_common",
    _repo_root / "setup" / "setup_common.py",
)
setup_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup_common)

_linux_spec = importlib.util.spec_from_file_location(
    "setup_linux",
    _repo_root / "setup" / "setup_linux.py",
)
# setup_linux imports setup_common from the same directory; load after patching path
import sys

sys.path.insert(0, str(_repo_root / "setup"))
setup_linux = importlib.util.module_from_spec(_linux_spec)
_linux_spec.loader.exec_module(setup_linux)


class TestInstallRequirementsFailure(unittest.TestCase):
    def test_install_requirements_inbulk_returns_false_on_pip_failure(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("some-package==1.0.0\n")
            req_path = tmp.name

        try:
            mock_proc = MagicMock()
            mock_proc.stdout = iter([])
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (
                "",
                "ERROR: HTTP error 403 while getting wheel",
            )

            with patch.object(setup_common.subprocess, "Popen", return_value=mock_proc):
                with patch.object(setup_common, "installed", return_value=False):
                    result = setup_common.install_requirements_inbulk(
                        req_path, show_stdout=False
                    )

            self.assertIs(result, False)
        finally:
            os.unlink(req_path)

    def test_install_requirements_inbulk_returns_true_on_success(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("some-package==1.0.0\n")
            req_path = tmp.name

        try:
            mock_proc = MagicMock()
            mock_proc.stdout = iter(["Requirement already satisfied: some-package\n"])
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = ("", "")

            with patch.object(setup_common.subprocess, "Popen", return_value=mock_proc):
                with patch.object(setup_common, "installed", return_value=False):
                    result = setup_common.install_requirements_inbulk(
                        req_path, show_stdout=False
                    )

            self.assertIs(result, True)
        finally:
            os.unlink(req_path)

    def test_install_requirements_inbulk_returns_false_when_file_missing(self):
        result = setup_common.install_requirements_inbulk(
            str(_repo_root / "no-such-requirements-file-xyz.txt"),
            show_stdout=False,
        )
        self.assertIs(result, False)

    def test_setup_linux_main_menu_exits_on_install_failure(self):
        with patch.object(setup_linux.setup_common, "check_repo_version"):
            with patch.object(setup_linux.setup_common, "install"):
                with patch.object(
                    setup_linux.setup_common,
                    "install_requirements_inbulk",
                    return_value=False,
                ):
                    with patch.object(
                        setup_linux.setup_common, "configure_accelerate"
                    ) as mock_accel:
                        with self.assertRaises(SystemExit) as ctx:
                            setup_linux.main_menu(
                                "requirements_ipex_xpu.txt",
                                show_stdout=False,
                                no_run_accelerate=False,
                            )
                        self.assertEqual(ctx.exception.code, 1)
                        mock_accel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
