"""Regression for issue #3499: native torch.xpu must not require IPEX."""

import importlib.util
import sys
import types
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


def _make_native_xpu_torch(version: str = "2.7.1+xpu"):
    """Minimal torch stand-in: XPU available, no CUDA, no IPEX."""
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = version
    torch_mod.cuda = MagicMock()
    torch_mod.cuda.is_available.return_value = False
    torch_mod.xpu = MagicMock()
    torch_mod.xpu.is_available.return_value = True
    torch_mod.xpu.device_count.return_value = 1
    torch_mod.xpu.device.return_value = 0
    torch_mod.xpu.get_device_name.return_value = "Intel Arc B580"
    props = MagicMock()
    props.total_memory = 12 * 1024 * 1024 * 1024
    props.max_compute_units = 160
    torch_mod.xpu.get_device_properties.return_value = props
    return torch_mod


class TestCheckTorchNativeXpu(unittest.TestCase):
    def test_check_torch_accepts_native_xpu_without_ipex(self):
        torch_mod = _make_native_xpu_torch()

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return torch_mod
            if name == "intel_extension_for_pytorch":
                raise ImportError("No module named 'intel_extension_for_pytorch'")
            return original_import(name, *args, **kwargs)

        original_import = __import__

        with patch.object(setup_common, "_check_hardware_toolkit"):
            with patch("builtins.__import__", side_effect=fake_import):
                major = setup_common.check_torch()

        self.assertEqual(major, 2)

    def test_log_gpu_info_reports_native_xpu_when_ipex_missing(self):
        torch_mod = _make_native_xpu_torch()
        logs = []

        def capture_info(msg, *args, **kwargs):
            logs.append(str(msg))

        with patch.object(setup_common.log, "info", side_effect=capture_info):
            with patch.object(setup_common.log, "warning") as mock_warn:
                with patch.dict(sys.modules, {"intel_extension_for_pytorch": None}):
                    # Force ImportError path for IPEX
                    real_import = __import__

                    def fake_import(name, *args, **kwargs):
                        if name == "intel_extension_for_pytorch":
                            raise ImportError("no ipex")
                        return real_import(name, *args, **kwargs)

                    with patch("builtins.__import__", side_effect=fake_import):
                        setup_common._log_gpu_info(torch_mod)

        joined = " ".join(logs)
        self.assertIn("native PyTorch XPU", joined)
        self.assertIn("Intel Arc B580", joined)
        mock_warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
