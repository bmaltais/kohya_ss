"""GH #3134: surface clear errors for missing/unwritable output directory
during training setup instead of uncaught PermissionError tracebacks.

Covers shared helpers and train_model early-return behavior on empty output_dir
and OSError during TOML write.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from kohya_gui import common_gui, textual_inversion_gui
from conftest import build_train_model_kwargs, mock_executor

_TI_FIXTURE = "test/config/TI-AdamW8bit-toml.json"
_TI_NUMERIC = (
    "lr_warmup_steps",
    "stop_text_encoder_training_pct",
    "main_process_port",
    "ip_noise_gamma",
    "ip_noise_gamma_random_strength",
    "huber_c",
    "huber_scale",
    "save_last_n_epochs",
    "save_last_n_epochs_state",
    "noise_offset_random_strength",
    "max_train_epochs",
)
_TI_STRINGS = (
    "log_with",
    "log_config",
    "lr_scheduler_type",
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_repo_type",
    "huggingface_repo_visibility",
    "huggingface_path_in_repo",
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    "dynamo_backend",
    "dynamo_mode",
    "extra_accelerate_launch_args",
)


class TestJoinConfigPath(unittest.TestCase):
    def test_joins_under_base(self):
        path = common_gui.join_config_path("/models/out", "config_lora-20250101.toml")
        self.assertEqual(path, os.path.join("/models/out", "config_lora-20250101.toml"))

    def test_empty_base_raises(self):
        with self.assertRaises(ValueError):
            common_gui.join_config_path("", "config_ti.toml")

    def test_empty_base_does_not_produce_root_absolute(self):
        # Regression: f"{''}/config_….toml" → "/config_….toml" on Unix
        with self.assertRaises(ValueError):
            common_gui.join_config_path("", "config_textual_inversion-x.toml")

    def test_whitespace_base_raises(self):
        with self.assertRaises(ValueError):
            common_gui.join_config_path("   ", "config.toml")


class TestRequireWritableDirectory(unittest.TestCase):
    def test_empty_rejects_with_required_message(self):
        with patch.object(common_gui, "output_message") as mock_msg:
            ok = common_gui.require_writable_directory(
                "", label="Output directory", headless=True
            )
        self.assertFalse(ok)
        msg = mock_msg.call_args.kwargs.get("msg") or mock_msg.call_args[0][0]
        self.assertIn("required", msg.lower())
        self.assertIn("output", msg.lower())

    def test_writable_dir_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(common_gui, "output_message") as mock_msg:
                ok = common_gui.require_writable_directory(tmp, headless=True)
            self.assertTrue(ok)
            mock_msg.assert_not_called()

    def test_creates_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "new_out")
            self.assertFalse(os.path.isdir(target))
            ok = common_gui.require_writable_directory(target, headless=True)
            self.assertTrue(ok)
            self.assertTrue(os.path.isdir(target))


class TestWriteTomlConfig(unittest.TestCase):
    def test_writes_successfully(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.toml")
            ok = common_gui.write_toml_config(path, {"a": 1}, headless=True)
            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(path))

    def test_oserror_surfaces_message_and_returns_false(self):
        with patch.object(common_gui, "output_message") as mock_msg:
            with patch("builtins.open", side_effect=PermissionError("denied")):
                ok = common_gui.write_toml_config(
                    "/nope/config.toml", {"a": 1}, headless=True
                )
        self.assertFalse(ok)
        mock_msg.assert_called()
        msg = mock_msg.call_args.kwargs.get("msg") or mock_msg.call_args[0][0]
        self.assertIn("config.toml", msg)
        self.assertTrue(
            any(k in msg.lower() for k in ("fail", "permission", "denied", "error")),
            msg=msg,
        )


class TestTextualInversionEmptyOutputDir(unittest.TestCase):
    def test_empty_output_dir_rejected_without_raising(self):
        kwargs = build_train_model_kwargs(
            textual_inversion_gui.train_model,
            _TI_FIXTURE,
            numeric_fixups=_TI_NUMERIC,
            string_overrides=_TI_STRINGS,
            overrides={
                "output_dir": "",
                "print_only": True,
                "token_string": "sks",
                "init_word": "person",
            },
        )
        mock_executor(textual_inversion_gui)
        with patch.object(
            textual_inversion_gui, "get_executable_path", return_value="accelerate"
        ):
            with patch.object(textual_inversion_gui, "output_message") as mock_msg:
                # require_writable_directory was imported into the TI module;
                # it calls common_gui.output_message unless we patch the
                # require helper's message channel via common_gui.
                with patch.object(common_gui, "output_message", mock_msg):
                    result = textual_inversion_gui.train_model(**kwargs)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        mock_msg.assert_called()
        msg = mock_msg.call_args.kwargs.get("msg") or mock_msg.call_args[0][0]
        self.assertIn("required", msg.lower())

    def test_toml_write_failure_returns_idle_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = build_train_model_kwargs(
                textual_inversion_gui.train_model,
                _TI_FIXTURE,
                numeric_fixups=_TI_NUMERIC,
                string_overrides=_TI_STRINGS,
                overrides={
                    "output_dir": tmp,
                    "print_only": True,
                    "token_string": "sks",
                    "init_word": "person",
                },
            )
            mock_executor(textual_inversion_gui)
            with patch.object(
                textual_inversion_gui,
                "get_executable_path",
                return_value="accelerate",
            ):
                with patch.object(
                    textual_inversion_gui,
                    "write_toml_config",
                    return_value=False,
                ):
                    result = textual_inversion_gui.train_model(**kwargs)

            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
