"""WD14 captioning (#3577): launch, exit codes, prefix/postfix, Keras preflight.

Covers kohya_gui.wd14_caption_gui.caption_images and check_keras_backend_ready.
No real tagger, GPU, or model weights.
"""

import inspect
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
        "caption_postfix": "",
        "onnx": True,
        "append_tags": False,
        "force_download": False,
        "caption_separator": ", ",
        "tag_replacement": "",
        "character_tag_expand": False,
        "use_rating_tags": False,
        "use_rating_tags_as_last_tag": False,
        "use_quality_tags": False,
        "use_quality_tags_as_last_tag": False,
        "character_tags_first": False,
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
        self.assertEqual(kwargs.get("cwd"), wd14_caption_gui.scriptdir)

    def test_success_applies_prefix_and_postfix(self):
        self.completed.returncode = 0
        _, mock_post, mock_log = self._run(
            always_first_tags="1girl",
            caption_postfix="masterpiece",
        )
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.kwargs["prefix"], "1girl")
        self.assertEqual(mock_post.call_args.kwargs["postfix"], "masterpiece")
        done_msgs = [
            c.args[0]
            for c in mock_log.info.call_args_list
            if c.args and "captioning done" in str(c.args[0]).lower()
        ]
        self.assertTrue(done_msgs, "expected success 'captioning done' log")

    def test_does_not_pass_always_first_tags_cli_flag(self):
        """Prefix is GUI post-process only — avoid double application via CLI."""
        mock_run, _, _ = self._run(always_first_tags="1girl")
        cmd = mock_run.call_args.args[0]
        self.assertNotIn("--always_first_tags", cmd)

    def test_quality_and_character_flags_passed(self):
        mock_run, _, _ = self._run(
            use_quality_tags=True,
            character_tags_first=True,
        )
        cmd = mock_run.call_args.args[0]
        self.assertIn("--use_quality_tags", cmd)
        self.assertIn("--character_tags_first", cmd)

    def test_caption_extension_normalizes_missing_dot(self):
        mock_run, mock_post, _ = self._run(caption_extension="wd14.txt")
        cmd = mock_run.call_args.args[0]
        ext_idx = cmd.index("--caption_extension")
        self.assertEqual(cmd[ext_idx + 1], ".wd14.txt")
        self.assertEqual(mock_post.call_args.kwargs["caption_file_ext"], ".wd14.txt")

    def test_nonzero_exit_skips_prefix_postfix_and_does_not_claim_done(self):
        self.completed.returncode = 1
        _, mock_post, mock_log = self._run(
            always_first_tags="1girl",
            caption_postfix="x",
        )
        mock_post.assert_not_called()
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

    def test_keras_preflight_blocks_launch_when_numpy2(self):
        fake_np = MagicMock()
        fake_np.__version__ = "2.2.6"
        with (
            patch.object(
                wd14_caption_gui,
                "check_keras_backend_ready",
                return_value="NumPy 2.x is installed; enable ONNX",
            ) as preflight,
            patch.object(wd14_caption_gui.subprocess, "run") as mock_run,
            patch.object(wd14_caption_gui, "log") as mock_log,
        ):
            wd14_caption_gui.caption_images(**_base_kwargs(onnx=False))
            preflight.assert_called_once()
            mock_run.assert_not_called()
            self.assertTrue(mock_log.error.called)


class TestCheckKerasBackendReady(unittest.TestCase):
    def test_numpy2_returns_actionable_message(self):
        import types

        np_mod = types.ModuleType("numpy")
        np_mod.__version__ = "2.1.0"
        with patch.dict(sys.modules, {"numpy": np_mod}):
            msg = wd14_caption_gui.check_keras_backend_ready()
        self.assertIsNotNone(msg)
        self.assertIn("NumPy", msg)
        self.assertRegex(msg, r"(?i)ONNX|1\.x|numpy")

    def test_numpy1_and_tf_ok_returns_none(self):
        import types

        np_mod = types.ModuleType("numpy")
        np_mod.__version__ = "1.26.4"
        tf_mod = types.ModuleType("tensorflow")
        with patch.dict(sys.modules, {"numpy": np_mod, "tensorflow": tf_mod}):
            msg = wd14_caption_gui.check_keras_backend_ready()
        self.assertIsNone(msg)


class TestWd14CaptionUiGuidance(unittest.TestCase):
    def test_module_documents_onnx_recommendation(self):
        source = inspect.getsource(wd14_caption_gui.gradio_wd14_caption_gui_tab)
        self.assertIn("ONNX", source)
        self.assertRegex(
            source,
            r"(?i)recommend|onnxruntime|TensorFlow|NumPy",
        )

    def test_ui_has_postfix_and_custom_extension(self):
        source = inspect.getsource(wd14_caption_gui.gradio_wd14_caption_gui_tab)
        self.assertIn("Postfix", source)
        self.assertIn("allow_custom_value=True", source)
        self.assertIn("caption_postfix", source)

    def test_repo_list_includes_eva02(self):
        self.assertTrue(
            any("eva02" in r for r in wd14_caption_gui.WD14_REPO_IDS),
            msg=wd14_caption_gui.WD14_REPO_IDS,
        )


class TestNumpyPinInManifests(unittest.TestCase):
    """Structural gate: project deps must pin NumPy away from 2.x (#3577 Phase 3)."""

    def test_requirements_txt_pins_numpy_below_2(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        text = (root / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r"(?m)^numpy\s*([=<>!~]=|[<>]=?).*",
            msg="requirements.txt must declare numpy",
        )
        # Must constrain below major 2
        self.assertTrue(
            any(
                line.strip().startswith("numpy")
                and ("<2" in line or "< 2" in line or "1.26" in line or "==1." in line)
                for line in text.splitlines()
            ),
            msg=f"numpy pin must exclude 2.x:\n{text}",
        )

    def test_pyproject_pins_numpy_below_2(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("numpy", text)
        self.assertTrue(
            "numpy" in text
            and (
                "numpy>=1.26.0,<2" in text
                or "numpy==1.26" in text
                or "numpy>=1.26,<2" in text
                or '"numpy>=1.26.0,<2"' in text
            ),
            msg="pyproject.toml must pin numpy <2",
        )


if __name__ == "__main__":
    unittest.main()
