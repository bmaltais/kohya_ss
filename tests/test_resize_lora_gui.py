"""Regression tests for GH issue #3529: expose --svd_lowrank_niter in the
Resize LoRA tab.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from kohya_gui import resize_lora_gui


class TestResizeLoraSvdLowrankNiter(unittest.TestCase):
    def setUp(self):
        self.model_file = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        self.model_file.close()
        self.addCleanup(os.unlink, self.model_file.name)

    def _run_and_capture_cmd(self, svd_lowrank_niter):
        with patch.object(resize_lora_gui.subprocess, "run") as mock_run:
            resize_lora_gui.resize_lora(
                model=self.model_file.name,
                new_rank=4,
                save_to=os.path.join(tempfile.gettempdir(), "out.safetensors"),
                save_precision="fp16",
                device="cpu",
                dynamic_method="None",
                dynamic_param="0.9",
                verbose=False,
                svd_lowrank_niter=svd_lowrank_niter,
            )
        self.assertTrue(mock_run.called)
        return mock_run.call_args[0][0]

    def test_includes_flag_with_given_value(self):
        run_cmd = self._run_and_capture_cmd(5)
        self.assertIn("--svd_lowrank_niter", run_cmd)
        idx = run_cmd.index("--svd_lowrank_niter")
        self.assertEqual(run_cmd[idx + 1], "5")

    def test_default_matches_backend_default(self):
        run_cmd = self._run_and_capture_cmd(2)
        idx = run_cmd.index("--svd_lowrank_niter")
        self.assertEqual(run_cmd[idx + 1], "2")

    def test_omits_flag_when_none(self):
        run_cmd = self._run_and_capture_cmd(None)
        self.assertNotIn("--svd_lowrank_niter", run_cmd)


if __name__ == "__main__":
    unittest.main()
