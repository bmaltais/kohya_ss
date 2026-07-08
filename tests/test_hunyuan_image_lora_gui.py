"""Regression tests for GH issue #3522: HunyuanImage-2.1 LoRA training
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

HUNYUAN_OVERRIDES = {
    "LoRA_type": "HunyuanImage-2.1",
    "hunyuan_image_checkbox": True,
    "hunyuan_text_encoder": "/models/qwen_2.5_vl_7b.safetensors",
    "hunyuan_byt5": "/models/byt5_small_glyphxl_fp16.safetensors",
    "hunyuan_vae": "/models/hunyuan_image_2.1_vae_fp16.safetensors",
    "hunyuan_discrete_flow_shift": 5.0,
    "hunyuan_model_prediction_type": "raw",
    "hunyuan_timestep_sampling": "sigma",
    "hunyuan_sigmoid_scale": 1.0,
    "hunyuan_attn_mode": "torch",
    "hunyuan_split_attn": False,
    "hunyuan_fp8_scaled": False,
    "hunyuan_fp8_vl": False,
    "hunyuan_text_encoder_cpu": False,
    "hunyuan_vae_chunk_size": 0,
}


class TestHunyuanImageLoraConfigOutput(unittest.TestCase):
    """End-to-end: train_model(print_only=True) writes a real TOML file we
    can inspect for the HunyuanImage-2.1 specific keys #3522 requires.
    """

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**HUNYUAN_OVERRIDES, **overrides},
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_network_module_is_hunyuan_image(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("network_module"), "networks.lora_hunyuan_image")

    def test_network_train_unet_only_is_forced(self):
        config = self._run_and_load_toml({})
        self.assertTrue(config.get("network_train_unet_only"))
        self.assertNotIn("network_train_text_encoder_only", config)

    def test_model_paths_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(
            config.get("text_encoder"), "/models/qwen_2.5_vl_7b.safetensors"
        )
        self.assertEqual(
            config.get("byt5"), "/models/byt5_small_glyphxl_fp16.safetensors"
        )
        self.assertEqual(
            config.get("vae"), "/models/hunyuan_image_2.1_vae_fp16.safetensors"
        )

    def test_flow_matching_params_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("discrete_flow_shift"), 5.0)
        self.assertEqual(config.get("model_prediction_type"), "raw")
        self.assertEqual(config.get("timestep_sampling"), "sigma")

    def test_incompatible_sd_args_are_excluded(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("max_token_length", config)
        self.assertNotIn("clip_skip", config)

    def test_script_selection_invokes_hunyuan_image_train_network(self):
        with patch.object(lora_gui, "print_command_and_toml") as mocked:
            kwargs = build_train_model_kwargs(
                lora_gui.train_model,
                FIXTURE,
                numeric_fixups=NUMERIC_FIXUPS,
                string_overrides=STRING_OVERRIDES,
                overrides=HUNYUAN_OVERRIDES,
            )
            mock_executor(lora_gui)
            lora_gui.train_model(**kwargs)
            self.assertTrue(mocked.called)
            run_cmd = mocked.call_args[0][0]
            self.assertTrue(
                any("hunyuan_image_train_network.py" in part for part in run_cmd)
            )


if __name__ == "__main__":
    unittest.main()
