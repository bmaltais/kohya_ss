"""Regression for issue #3499: Intel/XPU install must use native PyTorch XPU.

IPEX wheels from Intel's extension index return HTTP 403 after EOL.
--use-ipex must select a requirements file that does not pin IPEX.
"""

import importlib.util
import os
import unittest
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "setup_common",
    _repo_root / "setup" / "setup_common.py",
)
setup_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup_common)


def _requirements_text(filename: str) -> str:
    return (_repo_root / filename).read_text(encoding="utf-8")


class TestNativeXpuRequirements(unittest.TestCase):
    def test_linux_intel_gpu_requirements_file_is_native_xpu(self):
        filename = setup_common.get_linux_intel_gpu_requirements_file()
        self.assertEqual(filename, "requirements_ipex_xpu.txt")
        text = _requirements_text(filename)
        self.assertNotIn("intel-extension-for-pytorch", text)
        self.assertNotIn("pytorch-extension.intel.com", text)
        self.assertIn("download.pytorch.org/whl/xpu", text)
        self.assertIn("torch", text)
        self.assertIn("+xpu", text)

    def test_legacy_linux_ipex_requirements_file_does_not_pin_ipex(self):
        """Old name may still be referenced; it must not pull EOL IPEX wheels."""
        text = _requirements_text("requirements_linux_ipex.txt")
        self.assertNotIn("intel-extension-for-pytorch", text)
        self.assertNotIn("pytorch-extension.intel.com", text)


if __name__ == "__main__":
    unittest.main()
