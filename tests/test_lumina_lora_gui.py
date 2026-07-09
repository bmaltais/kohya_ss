"""Regression tests for GH issue #3360: Lumina Image 2.0 LoRA training
support in the GUI.
"""

import unittest
from unittest.mock import patch

from kohya_gui import lora_gui
from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
    mock_executor,
)

FIXTURE = "test/config/locon-AdamW8bit-toml.json"

NUMERIC_FIXUPS = (
    "max_train_steps",
    "max_train_epochs",
    "num_processes",
    "num_machines",
    "gradient_accumulation_steps",
    "seed",
    "vae_batch_size",
    "save_every_n_steps",
    "save_every_n_epochs",
    "save_last_n_steps",
    "save_last_n_steps_state",
    "save_last_n_epochs",
    "save_last_n_epochs_state",
    "lr_scheduler_num_cycles",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "max_data_loader_n_workers",
    "keep_tokens",
    "clip_skip",
    "max_token_length",
    "caption_dropout_every_n_epochs",
    "min_snr_gamma",
    "min_timestep",
    "max_timestep",
    "sample_every_n_steps",
    "sample_every_n_epochs",
    "gpu_ids",
    "main_process_port",
    "num_cpu_threads_per_process",
    "t5xxl_max_token_length",
    "block_lr_zero_threshold",
    "lumina_gemma2_max_token_length",
)
STRING_OVERRIDES = (
    "ae",
    "clip_l",
    "t5xxl",
    "sd3_clip_l",
    "sd3_t5xxl",
    "t5xxl_device",
    "t5xxl_dtype",
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
    "log_with",
    "log_config",
    "loss_type",
    "huber_schedule",
    "lr_scheduler_type",
    "dynamo_backend",
    "dynamo_mode",
    "extra_accelerate_launch_args",
    "network_weights",
    "model_prediction_type",
    "timestep_sampling",
    "train_blocks",
    "weighting_scheme",
    "in_dims",
    "lumina_system_prompt",
)

LUMINA_OVERRIDES = {
    "LoRA_type": "Lumina",
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


class TestLuminaLoraConfigOutput(unittest.TestCase):
    """End-to-end: train_model(print_only=True) writes a real TOML file we
    can inspect for the Lumina Image 2.0 specific keys #3360 requires.
    """

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**LUMINA_OVERRIDES, **overrides},
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_network_module_is_lora_lumina(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("network_module"), "networks.lora_lumina")

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

    def test_text_encoder_lora_is_not_forced_off(self):
        """Lumina can train Gemma2 LoRA unless the user opts into unet-only,
        so network_train_unet_only must not be forced True just because the
        Lumina checkbox is set (unlike HunyuanImage-2.1).
        """
        config = self._run_and_load_toml({"text_encoder_lr": 0.0001, "unet_lr": 0.0001})
        self.assertNotIn("network_train_unet_only", config)
        self.assertNotIn("network_train_text_encoder_only", config)

    def test_script_selection_invokes_lumina_train_network(self):
        with patch.object(lora_gui, "print_command_and_toml") as mocked:
            kwargs = build_train_model_kwargs(
                lora_gui.train_model,
                FIXTURE,
                numeric_fixups=NUMERIC_FIXUPS,
                string_overrides=STRING_OVERRIDES,
                overrides=LUMINA_OVERRIDES,
            )
            mock_executor(lora_gui)
            with patch.object(
                lora_gui, "get_executable_path", return_value="accelerate"
            ):
                lora_gui.train_model(**kwargs)
            self.assertTrue(mocked.called)
            run_cmd = mocked.call_args[0][0]
            self.assertTrue(any("lumina_train_network.py" in part for part in run_cmd))


if __name__ == "__main__":
    unittest.main()
