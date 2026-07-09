"""Regression tests for GH issue #3528: native LoHa/LoKr (networks.loha /
networks.lokr) support for SDXL and Anima in the LoRA tab.

These are distinct from the existing LyCORIS/LoHa and LyCORIS/LoKr options,
which use the third-party lycoris.kohya module. The native modules auto-detect
architecture (SDXL / Anima) and apply default targets/exclude_patterns.
"""

import unittest
from unittest.mock import patch

import gradio as gr

from kohya_gui import lora_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from conftest import (
    build_train_model_kwargs,
    mock_executor,
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

SDXL_LOHA = {
    "LoRA_type": "Kohya LoHa",
    "sdxl": True,
    "conv_dim": 0,
    "use_tucker": False,
    "rank_dropout": 0,
    "module_dropout": 0,
}
SDXL_LOKR = {
    "LoRA_type": "Kohya LoKr",
    "sdxl": True,
    "conv_dim": 0,
    "use_tucker": False,
    "factor": -1,
    "rank_dropout": 0,
    "module_dropout": 0,
}
ANIMA_BASE = {
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
    "conv_dim": 0,
    "use_tucker": False,
    "rank_dropout": 0,
    "module_dropout": 0,
}


class TestBuildNativeLohaLokrNetworkArgs(unittest.TestCase):
    """Pure-function unit tests for the native LoHa/LoKr network_args seam."""

    def test_loha_empty_when_optional_args_unset(self):
        result = lora_gui.build_native_loha_lokr_network_args(
            is_lokr=False,
            conv_dim=0,
            conv_alpha=1,
            use_tucker=False,
            factor=-1,
            rank_dropout=0,
            module_dropout=0,
        )
        self.assertEqual(result, "")

    def test_loha_includes_conv_and_tucker_when_set(self):
        result = lora_gui.build_native_loha_lokr_network_args(
            is_lokr=False,
            conv_dim=16,
            conv_alpha=8,
            use_tucker=True,
            factor=-1,
            rank_dropout=0.1,
            module_dropout=0.2,
        )
        self.assertIn("conv_dim=16", result)
        self.assertIn("conv_alpha=8", result)
        self.assertIn("use_tucker=True", result)
        self.assertIn("rank_dropout=0.1", result)
        self.assertIn("module_dropout=0.2", result)
        self.assertNotIn("factor=", result)

    def test_lokr_includes_factor_when_non_default(self):
        result = lora_gui.build_native_loha_lokr_network_args(
            is_lokr=True,
            conv_dim=0,
            conv_alpha=1,
            use_tucker=False,
            factor=4,
            rank_dropout=0,
            module_dropout=0,
        )
        self.assertIn("factor=4", result)
        self.assertNotIn("conv_dim=", result)


class TestKohyaLohaLokrConfigOutput(unittest.TestCase):
    """End-to-end: train_model(print_only=True) emits correct network_module
    and network_args for native LoHa/LoKr on SDXL and Anima.
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

    def test_sdxl_loha_sets_networks_loha(self):
        config = self._run_and_load_toml(SDXL_LOHA)
        self.assertEqual(config.get("network_module"), "networks.loha")

    def test_sdxl_lokr_sets_networks_lokr(self):
        config = self._run_and_load_toml(SDXL_LOKR)
        self.assertEqual(config.get("network_module"), "networks.lokr")

    def test_loha_does_not_use_lycoris_kohya(self):
        config = self._run_and_load_toml(SDXL_LOHA)
        self.assertNotEqual(config.get("network_module"), "lycoris.kohya")
        network_args = config.get("network_args") or []
        self.assertTrue(
            all("algo=" not in str(a) for a in network_args),
            f"native LoHa must not pass LyCORIS algo= args, got {network_args}",
        )

    def test_sdxl_loha_with_conv_dim_in_network_args(self):
        config = self._run_and_load_toml(
            {**SDXL_LOHA, "conv_dim": 16, "conv_alpha": 8, "use_tucker": True}
        )
        network_args = config.get("network_args") or []
        self.assertIn("conv_dim=16", network_args)
        self.assertIn("conv_alpha=8", network_args)
        self.assertIn("use_tucker=True", network_args)

    def test_sdxl_lokr_with_factor_in_network_args(self):
        config = self._run_and_load_toml({**SDXL_LOKR, "factor": 4})
        network_args = config.get("network_args") or []
        self.assertIn("factor=4", network_args)

    def test_anima_loha_sets_networks_loha_and_anima_script(self):
        with patch.object(lora_gui, "print_command_and_toml") as mocked:
            kwargs = build_train_model_kwargs(
                lora_gui.train_model,
                FIXTURE,
                numeric_fixups=NUMERIC_FIXUPS,
                string_overrides=STRING_OVERRIDES,
                overrides={**ANIMA_BASE, "LoRA_type": "Kohya LoHa"},
            )
            mock_executor(lora_gui)
            with patch.object(
                lora_gui, "get_executable_path", return_value="accelerate"
            ):
                lora_gui.train_model(**kwargs)
            self.assertTrue(mocked.called)
            run_cmd = mocked.call_args[0][0]
            self.assertTrue(
                any("anima_train_network.py" in part for part in run_cmd),
                f"expected anima_train_network.py in {run_cmd}",
            )
        config = self._run_and_load_toml({**ANIMA_BASE, "LoRA_type": "Kohya LoHa"})
        self.assertEqual(config.get("network_module"), "networks.loha")

    def test_anima_lokr_sets_networks_lokr(self):
        config = self._run_and_load_toml(
            {**ANIMA_BASE, "LoRA_type": "Kohya LoKr", "factor": -1}
        )
        self.assertEqual(config.get("network_module"), "networks.lokr")

    def test_loha_rejected_without_sdxl_or_anima(self):
        """Native LoHa/LoKr only support SDXL and Anima architectures."""
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={
                "LoRA_type": "Kohya LoHa",
                "sdxl": False,
                "anima_checkbox": False,
                "flux1_checkbox": False,
                "sd3_checkbox": False,
            },
        )
        mock_executor(lora_gui)
        with patch.object(lora_gui, "print_command_and_toml") as mocked:
            with patch.object(
                lora_gui, "get_executable_path", return_value="accelerate"
            ):
                result = lora_gui.train_model(**kwargs)
            mocked.assert_not_called()
            # train_model returns the train-button visibility tuple on abort
            self.assertIsNotNone(result)

    def test_loraplus_still_appended_for_loha(self):
        config = self._run_and_load_toml({**SDXL_LOHA, "loraplus_lr_ratio": 4.0})
        network_args = config.get("network_args") or []
        self.assertIn("loraplus_lr_ratio=4.0", network_args)


class TestKohyaLohaLokrDropdown(unittest.TestCase):
    """GUI surface: LoRA type dropdown must offer Kohya LoHa / Kohya LoKr."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = lora_gui.last_built_field_registry

    def test_dropdown_choices_include_kohya_loha_lokr(self):
        lora_type_comp = dict(self.field_registry).get("LoRA_type")
        self.assertIsNotNone(lora_type_comp)
        # Gradio may store choices as plain strings or (value, label) tuples.
        choices = [
            c[0] if isinstance(c, (list, tuple)) else c
            for c in list(lora_type_comp.choices)
        ]
        self.assertIn("Kohya LoHa", choices)
        self.assertIn("Kohya LoKr", choices)
        # Keep the existing third-party LyCORIS options distinct.
        self.assertIn("LyCORIS/LoHa", choices)
        self.assertIn("LyCORIS/LoKr", choices)


if __name__ == "__main__":
    unittest.main()
