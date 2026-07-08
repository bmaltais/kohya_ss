"""Regression tests for GH issue #3520: kohya_gui emitting args that
sd-scripts v0.11.1 silently ignores (LoRA+ ratios, stale `lowvram`).
"""

import unittest

from kohya_gui import lora_gui
from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_saved_json,
    run_train_model_and_load_toml,
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


class TestAppendLoraplusNetworkArgs(unittest.TestCase):
    """Pure-function unit test for the LoRA+ fix's core seam."""

    def test_appends_all_three_ratios_when_set(self):
        result = lora_gui.append_loraplus_network_args("preset=default", 16.0, 8.0, 4.0)
        self.assertEqual(
            result,
            "preset=default loraplus_lr_ratio=16.0"
            " loraplus_unet_lr_ratio=8.0 loraplus_text_encoder_lr_ratio=4.0",
        )

    def test_omits_zero_or_none_ratios(self):
        result = lora_gui.append_loraplus_network_args("preset=default", 0, None, 0.0)
        self.assertEqual(result, "preset=default")

    def test_appends_only_the_nonzero_ratio(self):
        result = lora_gui.append_loraplus_network_args("", 0, 8.0, 0)
        self.assertEqual(result, " loraplus_unet_lr_ratio=8.0")


class TestTrainModelConfigOutput(unittest.TestCase):
    """End-to-end: train_model(print_only=True) writes a real TOML file we
    can inspect for the specific keys #3520 flagged.
    """

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=overrides,
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_loraplus_ratios_flow_through_network_args_not_top_level(self):
        config = self._run_and_load_toml(
            {
                "loraplus_lr_ratio": 16.0,
                "loraplus_unet_lr_ratio": 8.0,
                "loraplus_text_encoder_lr_ratio": 4.0,
            }
        )
        self.assertNotIn("loraplus_lr_ratio", config)
        self.assertNotIn("loraplus_unet_lr_ratio", config)
        self.assertNotIn("loraplus_text_encoder_lr_ratio", config)
        network_args = config.get("network_args", [])
        self.assertIn("loraplus_lr_ratio=16.0", network_args)
        self.assertIn("loraplus_unet_lr_ratio=8.0", network_args)
        self.assertIn("loraplus_text_encoder_lr_ratio=4.0", network_args)

    def test_lowvram_never_appears_in_generated_config(self):
        config = self._run_and_load_toml({"lowvram": True})
        self.assertNotIn("lowvram", config)


class TestLoraTrainInpainting(unittest.TestCase):
    """GH issue #3527: inpainting model training support (SD1.5/SDXL).

    `--train_inpainting` must be forwarded to train_network.py/
    sdxl_train_network.py and must never be combined with
    `--cache_latents`/`--cache_latents_to_disk` since masks are generated
    randomly per step from the source image. It must also never leak into
    flux_train_network.py/sd3_train_network.py/hunyuan_image_train_network.py,
    which aren't in the supported script list.
    """

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=overrides,
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_train_inpainting_forwarded_and_cache_latents_dropped(self):
        config = self._run_and_load_toml(
            {
                "train_inpainting": True,
                "cache_latents": True,
                "cache_latents_to_disk": True,
            }
        )
        self.assertTrue(config.get("train_inpainting"))
        self.assertNotIn("cache_latents", config)
        self.assertNotIn("cache_latents_to_disk", config)

    def test_train_inpainting_dropped_for_flux_backend(self):
        # train_inpainting is only supported by train_network.py/
        # sdxl_train_network.py; flux_train_network.py must never receive it.
        config = self._run_and_load_toml(
            {
                "train_inpainting": True,
                "flux1_checkbox": True,
                "LoRA_type": "Flux1",
            }
        )
        self.assertNotIn("train_inpainting", config)

    def test_saved_json_config_reflects_forced_overrides(self):
        # The JSON training config saved via SaveConfigFile() must not
        # persist the pre-override checkbox state (train_inpainting=True
        # with cache_latents=True), which would produce an invalid combo
        # if the user reloads this saved config later.
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={
                "train_inpainting": True,
                "cache_latents": True,
                "cache_latents_to_disk": True,
            },
        )
        config = run_train_model_and_load_saved_json(lora_gui, kwargs)

        self.assertTrue(config.get("train_inpainting"))
        self.assertFalse(config.get("cache_latents"))
        self.assertFalse(config.get("cache_latents_to_disk"))


if __name__ == "__main__":
    unittest.main()
