"""Regression tests for GH issue #3523: Anima full fine-tune support in the
GUI (finetune tab → anima_train.py).
"""

import os
import tempfile
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
    "anima_discrete_flow_shift",
    "anima_sigmoid_scale",
    "anima_qwen3_max_token_length",
    "anima_t5_max_token_length",
    "anima_vae_chunk_size",
    "anima_compile_cache_size_limit",
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
    "anima_qwen3",
    "anima_vae",
    "anima_llm_adapter_path",
    "anima_t5_tokenizer_path",
    "anima_timestep_sampling",
    "anima_attn_mode",
    "anima_compile_backend",
    "anima_compile_mode",
    "anima_compile_dynamic",
)

ANIMA_OVERRIDES = {
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


class TestAnimaFinetuneConfigOutput(unittest.TestCase):
    """End-to-end: train_model(print_only=True) writes a real TOML file we
    can inspect for the Anima-specific keys #3523 requires.
    """

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**ANIMA_OVERRIDES, **overrides},
        )
        return run_train_model_and_load_toml(finetune_gui, kwargs)

    def test_script_selection_invokes_anima_train(self):
        with patch.object(finetune_gui, "print_command_and_toml") as mocked:
            kwargs = build_train_model_kwargs(
                finetune_gui.train_model,
                FIXTURE,
                numeric_fixups=NUMERIC_FIXUPS,
                string_overrides=STRING_OVERRIDES,
                overrides=ANIMA_OVERRIDES,
            )
            mock_executor(finetune_gui)
            with patch.object(
                finetune_gui, "get_executable_path", return_value="accelerate"
            ):
                finetune_gui.train_model(**kwargs)
            self.assertTrue(mocked.called)
            run_cmd = mocked.call_args[0][0]
            self.assertTrue(
                any("anima_train.py" in part for part in run_cmd),
                f"expected anima_train.py in {run_cmd}",
            )
            joined = " ".join(str(p) for p in run_cmd)
            self.assertNotIn("anima_train_network.py", joined)
            self.assertNotIn("flux_train.py", joined)
            self.assertNotIn("fine_tune.py", joined)

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

    def test_lora_only_args_are_excluded(self):
        """Full fine-tune must never emit LoRA network keys."""
        config = self._run_and_load_toml({})
        self.assertNotIn("network_module", config)
        self.assertNotIn("network_dim", config)
        self.assertNotIn("network_alpha", config)

    def test_attn_mode_omitted_when_default(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("attn_mode", config)

    def test_attn_mode_forwarded_when_non_default(self):
        config = self._run_and_load_toml({"anima_attn_mode": "xformers"})
        self.assertEqual(config.get("attn_mode"), "xformers")

    def test_train_inpainting_dropped_for_anima_backend(self):
        config = self._run_and_load_toml({"train_inpainting": True})
        self.assertNotIn("train_inpainting", config)

    def test_text_encoder_cache_forwarded(self):
        config = self._run_and_load_toml(
            {
                "anima_cache_text_encoder_outputs": True,
                "anima_cache_text_encoder_outputs_to_disk": True,
            }
        )
        self.assertTrue(config.get("cache_text_encoder_outputs"))
        self.assertTrue(config.get("cache_text_encoder_outputs_to_disk"))


class TestAnimaFinetuneAdvancedOptions(unittest.TestCase):
    """Advanced Anima options (compile, VAE 2D, timesteps) on the fine-tune tab."""

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**ANIMA_OVERRIDES, **overrides},
        )
        return run_train_model_and_load_toml(finetune_gui, kwargs)

    def test_qwen_image_vae_2d_forwarded_when_enabled(self):
        config = self._run_and_load_toml({"anima_qwen_image_vae_2d": True})
        self.assertTrue(config.get("qwen_image_vae_2d"))

    def test_compile_forwarded_with_options(self):
        config = self._run_and_load_toml(
            {
                "anima_compile": True,
                "anima_compile_backend": "inductor",
                "anima_compile_mode": "max-autotune",
                "anima_compile_dynamic": "true",
                "anima_compile_fullgraph": True,
                "anima_compile_cache_size_limit": 32,
            }
        )
        self.assertTrue(config.get("compile"))
        self.assertEqual(config.get("compile_backend"), "inductor")
        self.assertEqual(config.get("compile_mode"), "max-autotune")
        self.assertEqual(config.get("compile_dynamic"), "true")
        self.assertTrue(config.get("compile_fullgraph"))
        self.assertEqual(config.get("compile_cache_size_limit"), 32)

    def test_compile_options_omitted_when_compile_disabled(self):
        config = self._run_and_load_toml(
            {
                "anima_compile": False,
                "anima_compile_mode": "max-autotune",
                "anima_compile_fullgraph": True,
            }
        )
        self.assertNotIn("compile_mode", config)
        self.assertNotIn("compile_fullgraph", config)

    def test_compile_and_torch_compile_together_blocks_training(self):
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={
                **ANIMA_OVERRIDES,
                "anima_compile": True,
                "anima_torch_compile": True,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = dict(kwargs, output_dir=tmpdir, output_name="testout")
            mock_executor(finetune_gui)
            with patch.object(
                finetune_gui, "get_executable_path", return_value="accelerate"
            ):
                finetune_gui.train_model(**kwargs)
            toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
            self.assertEqual(toml_files, [])

    def test_show_timesteps_forwarded_for_anima(self):
        config = self._run_and_load_toml(
            {"show_timesteps": "console", "show_timesteps_resolution": "1024"}
        )
        self.assertEqual(config.get("show_timesteps"), "console")
        self.assertEqual(config.get("show_timesteps_resolution"), "1024")

    def test_blocks_to_swap_forwarded_for_anima(self):
        config = self._run_and_load_toml({"blocks_to_swap": 8})
        self.assertEqual(config.get("blocks_to_swap"), 8)

    def test_unsloth_offload_forwarded_when_enabled(self):
        config = self._run_and_load_toml({"anima_unsloth_offload_checkpointing": True})
        self.assertTrue(config.get("unsloth_offload_checkpointing"))


if __name__ == "__main__":
    unittest.main()
