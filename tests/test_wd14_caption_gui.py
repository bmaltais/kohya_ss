"""Phase 1 WD14 captioning (#3577): launch without accelerate, honor exit codes.

Covers kohya_gui.wd14_caption_gui.caption_images process invocation and
success/failure handling. No real tagger, GPU, or model weights.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kohya_gui import wd14_caption_gui


def _base_kwargs(**overrides):
    kwargs = {
        "train_data_dir": "/data/images",
        "caption_extension": ".txt",
        "batch_size": 1,
        "general_threshold": 0.35,
        "character_threshold": 0.35,
        "repo_id": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        "recursive": False,
        "max_data_loader_n_workers": 2,
        "debug": False,
        "undesired_tags": "",
        "frequency_tags": False,
        "always_first_tags": "",
        "onnx": True,
        "append_tags": False,
        "force_download": False,
        "caption_separator": ", ",
        "tag_replacement": "",
        "character_tag_expand": False,
        "use_rating_tags": False,
        "use_rating_tags_as_last_tag": False,
        "remove_underscore": True,
        "thresh": 0.35,
    }
    kwargs.update(overrides)
    return kwargs


class TestWd14CaptionImagesLaunch(unittest.TestCase):
    def setUp(self):
        self.env = {"PATH": "test"}
        self.completed = SimpleNamespace(returncode=0)

    def _run(self, **overrides):
        with (
            patch.object(wd14_caption_gui.os.path, "exists", return_value=True),
            patch.object(wd14_caption_gui, "setup_environment", return_value=self.env),
            patch.object(
                wd14_caption_gui.subprocess,
                "run",
                return_value=self.completed,
            ) as mock_run,
            patch.object(wd14_caption_gui, "add_pre_postfix") as mock_prefix,
            patch.object(wd14_caption_gui, "log") as mock_log,
        ):
            wd14_caption_gui.caption_images(**_base_kwargs(**overrides))
            return mock_run, mock_prefix, mock_log

    def test_launches_with_sys_executable_not_accelerate(self):
        mock_run, _, _ = self._run()
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertNotIn("accelerate", " ".join(str(p) for p in cmd).lower())
        self.assertTrue(
            str(cmd[1])
            .replace("\\", "/")
            .endswith("sd-scripts/finetune/tag_images_by_wd14_tagger.py"),
            msg=cmd[1],
        )
        self.assertIn("--onnx", cmd)

    def test_subprocess_uses_shell_false_and_project_root_cwd(self):
        mock_run, _, _ = self._run()
        kwargs = mock_run.call_args.kwargs
        self.assertIs(kwargs.get("shell"), False)
        self.assertEqual(kwargs.get("env"), self.env)
        # Project root so default wd14_tagger_model/ stays repo-local
        self.assertEqual(kwargs.get("cwd"), wd14_caption_gui.scriptdir)

    def test_success_runs_prefix_and_logs_done(self):
        self.completed.returncode = 0
        _, mock_prefix, mock_log = self._run(always_first_tags="1girl")
        mock_prefix.assert_called_once()
        self.assertEqual(mock_prefix.call_args.kwargs["prefix"], "1girl")
        done_msgs = [
            c.args[0]
            for c in mock_log.info.call_args_list
            if c.args and "captioning done" in str(c.args[0]).lower()
        ]
        self.assertTrue(done_msgs, "expected success 'captioning done' log")

    def test_nonzero_exit_skips_prefix_and_does_not_claim_done(self):
        self.completed.returncode = 1
        _, mock_prefix, mock_log = self._run()
        mock_prefix.assert_not_called()
        done_msgs = [
            c.args[0]
            for c in mock_log.info.call_args_list
            if c.args and "captioning done" in str(c.args[0]).lower()
        ]
        self.assertFalse(done_msgs, f"must not claim done on failure: {done_msgs}")
        error_calls = [c.args[0] for c in mock_log.error.call_args_list if c.args]
        self.assertTrue(
            any(
                "fail" in str(m).lower() or "exit" in str(m).lower()
                for m in error_calls
            ),
            msg=error_calls,
        )

    def test_missing_train_dir_does_not_launch(self):
        with patch.object(wd14_caption_gui.subprocess, "run") as mock_run:
            wd14_caption_gui.caption_images(**_base_kwargs(train_data_dir=""))
            mock_run.assert_not_called()


class TestWd14CaptionUiGuidance(unittest.TestCase):
    def test_module_documents_onnx_recommendation(self):
        """UI copy is source-level; avoid building full Gradio tree in unit tests."""
        import inspect

        source = inspect.getsource(wd14_caption_gui.gradio_wd14_caption_gui_tab)
        self.assertIn("ONNX", source)
        self.assertRegex(
            source,
            r"(?i)recommend|onnxruntime|TensorFlow|NumPy",
        )


if __name__ == "__main__":
    unittest.main()
