"""Regression tests for GH issue #3521: Lumina Image 2.0 full fine-tune
support in the GUI (finetune tab → lumina_train.py).
"""

import unittest
from unittest.mock import patch

from kohya_gui import finetune_gui
from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
    mock_executor,
)

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
    "lumina_discrete_flow_shift",
    "lumina_sigmoid_scale",
    "lumina_gemma2_max_token_length",
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
    "lumina_gemma2",
    "lumina_ae",
    "lumina_model_prediction_type",
    "lumina_timestep_sampling",
    "lumina_system_prompt",
)

LUMINA_OVERRIDES = {
    "lumina_checkbox": True,
    "lumina_gemma2": "/models/gemma-2-2b.safetensors",
    "lumina_ae": "/models/ae.safetensors",
    "lumina_discrete_flow_shift": 6.0,
    "lumina_model_prediction_type": "raw",
    "lumina_timestep_sampling": "nextdit_shift",
    "lumina_sigmoid_scale": 1.0,
    "lumina_gemma2_max_token_length": 256,
    "lumina_system_prompt": (
        "You are an assistant designed to generate high-quality images "
        "based on user prompts."
    ),
    "lumina_use_flash_attn": False,
    "lumina_use_sage_attn": False,
    "lumina_cache_text_encoder_outputs": True,
    "lumina_cache_text_encoder_outputs_to_disk": False,
}


class TestLuminaFinetuneConfigOutput(unittest.TestCase):
    """End-to-end: train_model(print_only=True) writes a real TOML file we
    can inspect for the Lumina Image 2.0 fine-tune keys #3521 requires.
    """

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**LUMINA_OVERRIDES, **overrides},
        )
        return run_train_model_and_load_toml(finetune_gui, kwargs)

    def test_script_selection_invokes_lumina_train(self):
        with patch.object(finetune_gui, "print_command_and_toml") as mocked:
            kwargs = build_train_model_kwargs(
                finetune_gui.train_model,
                FIXTURE,
                numeric_fixups=NUMERIC_FIXUPS,
                string_overrides=STRING_OVERRIDES,
                overrides=LUMINA_OVERRIDES,
            )
            mock_executor(finetune_gui)
            with patch.object(
                finetune_gui, "get_executable_path", return_value="accelerate"
            ):
                finetune_gui.train_model(**kwargs)
            self.assertTrue(mocked.called)
            run_cmd = mocked.call_args[0][0]
            self.assertTrue(
                any("lumina_train.py" in part for part in run_cmd),
                f"expected lumina_train.py in {run_cmd}",
            )
            joined = " ".join(str(p) for p in run_cmd)
            self.assertNotIn("lumina_train_network.py", joined)
            self.assertNotIn("flux_train.py", joined)
            self.assertNotIn("fine_tune.py", joined)

    def test_model_paths_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("gemma2"), "/models/gemma-2-2b.safetensors")
        self.assertEqual(config.get("ae"), "/models/ae.safetensors")

    def test_flow_matching_params_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("discrete_flow_shift"), 6.0)
        self.assertEqual(config.get("model_prediction_type"), "raw")
        self.assertEqual(config.get("timestep_sampling"), "nextdit_shift")
        self.assertEqual(config.get("sigmoid_scale"), 1.0)

    def test_system_prompt_and_token_length_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("gemma2_max_token_length"), 256)
        self.assertIn("high-quality images", config.get("system_prompt", ""))

    def test_text_encoder_cache_flags_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertTrue(config.get("cache_text_encoder_outputs"))
        self.assertNotIn("cache_text_encoder_outputs_to_disk", config)

    def test_attention_flags_forwarded_when_set(self):
        config = self._run_and_load_toml(
            {"lumina_use_flash_attn": True, "lumina_use_sage_attn": True}
        )
        self.assertTrue(config.get("use_flash_attn"))
        self.assertTrue(config.get("use_sage_attn"))

    def test_incompatible_sd_args_are_excluded(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("max_token_length", config)
        self.assertNotIn("clip_skip", config)

    def test_lora_only_args_are_excluded(self):
        """Full fine-tune must never emit LoRA network keys."""
        config = self._run_and_load_toml({})
        self.assertNotIn("network_module", config)
        self.assertNotIn("network_dim", config)
        self.assertNotIn("network_alpha", config)

    def test_train_inpainting_dropped_for_lumina_backend(self):
        config = self._run_and_load_toml({"train_inpainting": True})
        self.assertNotIn("train_inpainting", config)

    def test_show_timesteps_forwarded_for_lumina(self):
        config = self._run_and_load_toml(
            {"show_timesteps": "console", "show_timesteps_resolution": "1024"}
        )
        self.assertEqual(config.get("show_timesteps"), "console")
        self.assertEqual(config.get("show_timesteps_resolution"), "1024")

    def test_blocks_to_swap_forwarded_for_lumina(self):
        config = self._run_and_load_toml({"blocks_to_swap": 8})
        self.assertEqual(config.get("blocks_to_swap"), 8)


if __name__ == "__main__":
    unittest.main()
