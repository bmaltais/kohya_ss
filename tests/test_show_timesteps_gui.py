"""Regression tests for GH issue #3530: expose --show_timesteps /
--show_timesteps_resolution (shared DiT timestep-sampling visualization
flags from sd-scripts library/args.py add_dit_training_arguments) in the
GUI's advanced training options, and only emit them for DiT families
(FLUX / SD3), never for SD1.5/SDXL.
"""

import unittest

from kohya_gui import lora_gui
from conftest import build_train_model_kwargs, run_train_model_and_load_toml

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
    "show_timesteps",
    "show_timesteps_resolution",
)


class TestShowTimestepsFlow(unittest.TestCase):
    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=overrides,
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_flux1_emits_show_timesteps(self):
        config = self._run_and_load_toml(
            {
                "flux1_checkbox": True,
                "LoRA_type": "Flux1",
                "show_timesteps": "console",
                "show_timesteps_resolution": "512,768",
            }
        )
        self.assertEqual(config.get("show_timesteps"), "console")
        self.assertEqual(config.get("show_timesteps_resolution"), "512,768")

    def test_sd3_emits_show_timesteps(self):
        config = self._run_and_load_toml(
            {
                "sd3_checkbox": True,
                "show_timesteps": "image",
                "show_timesteps_resolution": "1024",
            }
        )
        self.assertEqual(config.get("show_timesteps"), "image")
        self.assertEqual(config.get("show_timesteps_resolution"), "1024")

    def test_omitted_when_flag_left_blank(self):
        config = self._run_and_load_toml(
            {
                "flux1_checkbox": True,
                "LoRA_type": "Flux1",
                "show_timesteps": "",
            }
        )
        self.assertNotIn("show_timesteps", config)

    def test_never_emitted_for_non_dit_families(self):
        config = self._run_and_load_toml(
            {
                "flux1_checkbox": False,
                "sd3_checkbox": False,
                "show_timesteps": "console",
            }
        )
        self.assertNotIn("show_timesteps", config)
        self.assertNotIn("show_timesteps_resolution", config)


if __name__ == "__main__":
    unittest.main()
