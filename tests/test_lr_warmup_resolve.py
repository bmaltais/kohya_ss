"""Regression tests for GH issue #3455: string/None lr_warmup values must not
raise TypeError when computing lr_warmup_steps in train_model paths.

Covers the shared resolve_lr_warmup_steps helper and fine-tune train_model
integration with string-typed warmup controls.
"""

import unittest
from unittest.mock import patch

from kohya_gui.common_gui import resolve_lr_warmup_steps
from kohya_gui import finetune_gui
from kohya_gui.class_basic_training import BasicTraining
from conftest import build_train_model_kwargs, run_train_model_and_load_toml

FIXTURE = "test/config/finetune-AdamW-toml.json"

NUMERIC_FIXUPS = (
    "lr_warmup_steps",
    "huber_scale",
    "save_last_n_epochs",
    "save_last_n_epochs_state",
    "logit_mean",
    "logit_std",
    "mode_scale",
    "sd3_text_encoder_batch_size",
    "t5xxl_max_token_length",
    "guidance_scale",
    "blocks_to_swap",
    "single_blocks_to_swap",
    "double_blocks_to_swap",
    "discrete_flow_shift",
)
STRING_OVERRIDES = (
    "ae",
    "clip_l",
    "clip_g",
    "t5xxl",
    "flux1_clip_l",
    "flux1_t5xxl",
    "t5xxl_device",
    "t5xxl_dtype",
    "log_config",
    "lr_scheduler_type",
    "model_prediction_type",
    "timestep_sampling",
    "weighting_scheme",
)


class TestResolveLrWarmupSteps(unittest.TestCase):
    """Unit tests for resolve_lr_warmup_steps — pure precedence + coercion."""

    def test_numeric_percent_yields_fraction(self):
        self.assertEqual(resolve_lr_warmup_steps(10, 0), 0.1)
        self.assertEqual(resolve_lr_warmup_steps(10.0, 0), 0.1)

    def test_numeric_override_takes_precedence(self):
        with patch("kohya_gui.common_gui.log") as mock_log:
            result = resolve_lr_warmup_steps(10, 50)
        self.assertEqual(result, 50)
        mock_log.warning.assert_called_once()

    def test_override_without_percent_no_warning(self):
        with patch("kohya_gui.common_gui.log") as mock_log:
            result = resolve_lr_warmup_steps(0, 50)
        self.assertEqual(result, 50)
        mock_log.warning.assert_not_called()

    def test_both_zero(self):
        self.assertEqual(resolve_lr_warmup_steps(0, 0), 0)

    def test_string_percent_coerced(self):
        self.assertEqual(resolve_lr_warmup_steps("10", 0), 0.1)
        self.assertEqual(resolve_lr_warmup_steps("10", "0"), 0.1)

    def test_string_override_coerced(self):
        with patch("kohya_gui.common_gui.log"):
            self.assertEqual(resolve_lr_warmup_steps("10", "100"), 100)

    def test_empty_and_none_treated_as_zero(self):
        self.assertEqual(resolve_lr_warmup_steps("", None), 0)
        self.assertEqual(resolve_lr_warmup_steps(None, ""), 0)
        self.assertEqual(resolve_lr_warmup_steps(None, None), 0)

    def test_invalid_string_falls_back_to_zero(self):
        self.assertEqual(resolve_lr_warmup_steps("not-a-number", 0), 0)
        self.assertEqual(resolve_lr_warmup_steps(10, "bad"), 0.1)


class TestFinetuneStringLrWarmup(unittest.TestCase):
    """Integration: finetune train_model must accept string lr_warmup."""

    def _kwargs(self, **overrides):
        return build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=overrides,
        )

    def test_string_percent_writes_fraction(self):
        kwargs = self._kwargs(lr_warmup="10", lr_warmup_steps=0)
        config = run_train_model_and_load_toml(finetune_gui, kwargs)
        self.assertAlmostEqual(config.get("lr_warmup_steps"), 0.1)

    def test_string_percent_with_numeric_override(self):
        kwargs = self._kwargs(lr_warmup="10", lr_warmup_steps=100)
        config = run_train_model_and_load_toml(finetune_gui, kwargs)
        self.assertEqual(config.get("lr_warmup_steps"), 100)

    def test_numeric_inputs_unchanged(self):
        kwargs = self._kwargs(lr_warmup=10, lr_warmup_steps=0)
        config = run_train_model_and_load_toml(finetune_gui, kwargs)
        self.assertAlmostEqual(config.get("lr_warmup_steps"), 0.1)


class TestBasicTrainingLrWarmupDefault(unittest.TestCase):
    def test_default_lr_warmup_value_is_numeric(self):
        # Default used by finetune (no override) must not be a str.
        import inspect

        sig = inspect.signature(BasicTraining.__init__)
        default = sig.parameters["lr_warmup_value"].default
        self.assertIsInstance(default, (int, float))
        self.assertNotIsInstance(default, bool)
        self.assertEqual(default, 0)


if __name__ == "__main__":
    unittest.main()
