"""Regression tests for GH issue #3487: Anima LoRA training support in the
GUI.
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
)

ANIMA_OVERRIDES = {
    "LoRA_type": "Anima",
    "anima_checkbox": True,
    "anima_qwen3": "/models/qwen3_0.6b.safetensors",
    "anima_vae": "/models/qwen_image_vae_fp16.safetensors",
    "anima_llm_adapter_path": "",
    "anima_t5_tokenizer_path": "",
    "anima_discrete_flow_shift": 1.0,
    "anima_timestep_sampling": "sigmoid",
    "anima_sigmoid_scale": 1.0,
    "anima_qwen3_max_token_length": 512,
    "anima_t5_max_token_length": 512,
    "anima_attn_mode": "torch",
    "anima_split_attn": False,
    "anima_vae_chunk_size": 0,
    "anima_vae_disable_cache": False,
    "anima_unsloth_offload_checkpointing": False,
}


class TestAnimaLoraConfigOutput(unittest.TestCase):
    """End-to-end: train_model(print_only=True) writes a real TOML file we
    can inspect for the Anima specific keys #3487 requires.
    """

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**ANIMA_OVERRIDES, **overrides},
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_network_module_is_lora_anima(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("network_module"), "networks.lora_anima")

    def test_model_paths_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("qwen3"), "/models/qwen3_0.6b.safetensors")
        self.assertEqual(config.get("vae"), "/models/qwen_image_vae_fp16.safetensors")

    def test_optional_model_paths_are_omitted_when_blank(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("llm_adapter_path", config)
        self.assertNotIn("t5_tokenizer_path", config)

    def test_optional_model_paths_are_forwarded_when_set(self):
        config = self._run_and_load_toml(
            {
                "anima_llm_adapter_path": "/models/llm_adapter.safetensors",
                "anima_t5_tokenizer_path": "/models/t5_tokenizer",
            }
        )
        self.assertEqual(
            config.get("llm_adapter_path"), "/models/llm_adapter.safetensors"
        )
        self.assertEqual(config.get("t5_tokenizer_path"), "/models/t5_tokenizer")

    def test_flow_matching_params_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("discrete_flow_shift"), 1.0)
        self.assertEqual(config.get("timestep_sampling"), "sigmoid")
        self.assertEqual(config.get("sigmoid_scale"), 1.0)

    def test_token_length_params_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("qwen3_max_token_length"), 512)
        self.assertEqual(config.get("t5_max_token_length"), 512)

    def test_incompatible_sd_args_are_excluded(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("max_token_length", config)
        self.assertNotIn("clip_skip", config)

    def test_text_encoder_lora_is_not_forced_off(self):
        """Unlike HunyuanImage-2.1, Anima supports training the (Qwen3) text
        encoder LoRA, so network_train_unet_only must not be forced True
        just because the Anima checkbox is set.
        """
        config = self._run_and_load_toml({"text_encoder_lr": 0.0001, "unet_lr": 0.0001})
        self.assertNotIn("network_train_unet_only", config)
        self.assertNotIn("network_train_text_encoder_only", config)

    def test_attn_mode_omitted_when_default(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("attn_mode", config)

    def test_attn_mode_forwarded_when_non_default(self):
        config = self._run_and_load_toml({"anima_attn_mode": "xformers"})
        self.assertEqual(config.get("attn_mode"), "xformers")

    def test_script_selection_invokes_anima_train_network(self):
        with patch.object(lora_gui, "print_command_and_toml") as mocked:
            kwargs = build_train_model_kwargs(
                lora_gui.train_model,
                FIXTURE,
                numeric_fixups=NUMERIC_FIXUPS,
                string_overrides=STRING_OVERRIDES,
                overrides=ANIMA_OVERRIDES,
            )
            mock_executor(lora_gui)
            lora_gui.train_model(**kwargs)
            self.assertTrue(mocked.called)
            run_cmd = mocked.call_args[0][0]
            self.assertTrue(any("anima_train_network.py" in part for part in run_cmd))


if __name__ == "__main__":
    unittest.main()
