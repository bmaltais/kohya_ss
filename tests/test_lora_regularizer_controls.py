"""Regression tests for GH issue #3389: restore Scale weight norms, Network
dropout, Rank dropout, and Module dropout for modern native LoRA types.

sd-scripts networks.lora_flux / lora_anima / lora_lumina /
lora_hunyuan_image accept neuron_dropout (top-level --network_dropout),
rank_dropout / module_dropout (network kwargs), and max-norm via
apply_max_norm_regularization (--scale_weight_norms). The GUI previously
hid those four controls for Flux1 and peers, and omitted rank/module
dropout from Flux1 network_args even when non-zero.

Flux1 OFT (networks.oft_flux) does not implement these hooks — controls
must stay hidden for that type.
"""

import unittest

from kohya_gui import lora_gui
from conftest import (
    build_train_model_kwargs,
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
    "model_type",
    "model_prediction_type",
    "timestep_sampling",
    "train_blocks",
    "weighting_scheme",
    "in_dims",
)

# Types whose sd-scripts network modules implement the four regularizers.
MODERN_NATIVE_TYPES = ("Flux1", "Anima", "Lumina", "HunyuanImage-2.1")


class TestRegularizerVisibilityAllowLists(unittest.TestCase):
    """Allow-lists drive Basic-accordion slider visibility per LoRA type."""

    def test_modern_native_types_show_all_four_controls(self):
        for lora_type in MODERN_NATIVE_TYPES:
            with self.subTest(lora_type=lora_type):
                vis = lora_gui.regularizer_controls_visible(lora_type)
                self.assertTrue(vis["scale_weight_norms"], lora_type)
                self.assertTrue(vis["network_dropout"], lora_type)
                self.assertTrue(vis["rank_dropout"], lora_type)
                self.assertTrue(vis["module_dropout"], lora_type)

    def test_standard_still_shows_all_four(self):
        vis = lora_gui.regularizer_controls_visible("Standard")
        self.assertTrue(vis["scale_weight_norms"])
        self.assertTrue(vis["network_dropout"])
        self.assertTrue(vis["rank_dropout"])
        self.assertTrue(vis["module_dropout"])

    def test_flux1_oft_hides_all_four(self):
        # networks.oft_flux accepts neuron_dropout in the signature but never
        # uses it; no rank/module dropout or apply_max_norm either.
        vis = lora_gui.regularizer_controls_visible("Flux1 OFT")
        self.assertFalse(vis["scale_weight_norms"])
        self.assertFalse(vis["network_dropout"])
        self.assertFalse(vis["rank_dropout"])
        self.assertFalse(vis["module_dropout"])


class TestAppendRankModuleDropoutNetworkArgs(unittest.TestCase):
    """Pure-function seam for rank/module dropout network kwargs."""

    def test_appends_nonzero_values(self):
        result = lora_gui.append_rank_module_dropout_network_args(
            "", rank_dropout=0.1, module_dropout=0.2
        )
        self.assertIn("rank_dropout=0.1", result)
        self.assertIn("module_dropout=0.2", result)

    def test_omits_zero_and_none(self):
        result = lora_gui.append_rank_module_dropout_network_args(
            "split_qkv=True", rank_dropout=0, module_dropout=None
        )
        self.assertEqual(result, "split_qkv=True")
        self.assertNotIn("rank_dropout", result)
        self.assertNotIn("module_dropout", result)

    def test_preserves_existing_args_prefix(self):
        result = lora_gui.append_rank_module_dropout_network_args(
            " train_t5xxl=True", rank_dropout=0.15, module_dropout=0
        )
        self.assertTrue(result.startswith(" train_t5xxl=True"))
        self.assertIn("rank_dropout=0.15", result)
        self.assertNotIn("module_dropout", result)


class TestModernLoraRegularizerConfigOutput(unittest.TestCase):
    """train_model(print_only=True) must wire top-level + network_args correctly."""

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=overrides,
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_flux1_nonzero_rank_module_dropout_in_network_args(self):
        config = self._run_and_load_toml(
            {
                "flux1_checkbox": True,
                "LoRA_type": "Flux1",
                "rank_dropout": 0.1,
                "module_dropout": 0.2,
            }
        )
        self.assertEqual(config.get("network_module"), "networks.lora_flux")
        network_args = config.get("network_args") or []
        self.assertIn("rank_dropout=0.1", network_args)
        self.assertIn("module_dropout=0.2", network_args)

    def test_flux1_nonzero_network_dropout_and_scale_weight_norms_top_level(self):
        config = self._run_and_load_toml(
            {
                "flux1_checkbox": True,
                "LoRA_type": "Flux1",
                "network_dropout": 0.3,
                "scale_weight_norms": 1.0,
            }
        )
        self.assertEqual(config.get("network_dropout"), 0.3)
        self.assertEqual(config.get("scale_weight_norms"), 1.0)

    def test_flux1_defaults_omit_dropout_and_scale_weight_overrides(self):
        config = self._run_and_load_toml(
            {
                "flux1_checkbox": True,
                "LoRA_type": "Flux1",
                "rank_dropout": 0,
                "module_dropout": 0,
                "network_dropout": 0,
                "scale_weight_norms": 0,
            }
        )
        self.assertNotIn("network_dropout", config)
        self.assertNotIn("scale_weight_norms", config)
        network_args = config.get("network_args") or []
        self.assertNotIn("rank_dropout=0", network_args)
        self.assertNotIn("module_dropout=0", network_args)
        for arg in network_args:
            self.assertFalse(
                arg.startswith("rank_dropout="),
                f"unexpected rank_dropout in default config: {arg}",
            )
            self.assertFalse(
                arg.startswith("module_dropout="),
                f"unexpected module_dropout in default config: {arg}",
            )

    def test_anima_rank_module_dropout_in_network_args(self):
        config = self._run_and_load_toml(
            {
                "anima_checkbox": True,
                "LoRA_type": "Anima",
                "anima_qwen3": "/models/qwen3.safetensors",
                "anima_vae": "/models/vae.safetensors",
                "rank_dropout": 0.1,
                "module_dropout": 0.2,
            }
        )
        self.assertEqual(config.get("network_module"), "networks.lora_anima")
        network_args = config.get("network_args") or []
        self.assertIn("rank_dropout=0.1", network_args)
        self.assertIn("module_dropout=0.2", network_args)

    def test_lumina_rank_module_dropout_in_network_args(self):
        config = self._run_and_load_toml(
            {
                "lumina_checkbox": True,
                "LoRA_type": "Lumina",
                "lumina_gemma2": "/models/gemma2.safetensors",
                "lumina_ae": "/models/ae.safetensors",
                "rank_dropout": 0.1,
                "module_dropout": 0.2,
            }
        )
        self.assertEqual(config.get("network_module"), "networks.lora_lumina")
        network_args = config.get("network_args") or []
        self.assertIn("rank_dropout=0.1", network_args)
        self.assertIn("module_dropout=0.2", network_args)

    def test_hunyuan_image_rank_module_dropout_in_network_args(self):
        config = self._run_and_load_toml(
            {
                "hunyuan_image_checkbox": True,
                "LoRA_type": "HunyuanImage-2.1",
                "rank_dropout": 0.1,
                "module_dropout": 0.2,
            }
        )
        self.assertEqual(config.get("network_module"), "networks.lora_hunyuan_image")
        network_args = config.get("network_args") or []
        self.assertIn("rank_dropout=0.1", network_args)
        self.assertIn("module_dropout=0.2", network_args)


if __name__ == "__main__":
    unittest.main()
