"""Regression tests for GH issue #3525: Anima ControlNet-LLLite training tab.

Covers command generation against anima_train_control_net_lllite.py, LLLite-
specific args (target_layers, lllite_use_aspp, conditioning dataset), and the
FIELD_REGISTRY / dict-adapter wiring pattern (GH #3543).
"""

import inspect
import os
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
import toml

from kohya_gui import anima_lllite_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from conftest import mock_executor


def _default_kwargs(overrides=None):
    sig = inspect.signature(anima_lllite_gui.train_model)
    params = [p for p in sig.parameters if p not in ("headless", "print_only")]

    defaults = {
        "pretrained_model_name_or_path": "/models/anima_dit.safetensors",
        "output_dir": "",
        "output_name": "test_anima_lllite",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "training_comment": "",
        "no_metadata": False,
        "train_data_dir": "/data/train",
        "conditioning_data_dir": "/data/cond",
        "dataset_config": "",
        "resolution": "1024",
        "enable_bucket": True,
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "train_batch_size": 1,
        "max_train_epochs": 10,
        "max_train_steps": 0,
        "caption_extension": ".txt",
        "cache_latents": True,
        "cache_latents_to_disk": False,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": False,
        "seed": 0,
        "gradient_accumulation_steps": 1,
        "anima_qwen3": "/models/qwen3_0.6b.safetensors",
        "anima_vae": "/models/qwen_image_vae_fp16.safetensors",
        "anima_llm_adapter_path": "",
        "anima_t5_tokenizer_path": "",
        "anima_discrete_flow_shift": 3.0,
        "anima_timestep_sampling": "shift",
        "anima_sigmoid_scale": 1.0,
        "anima_qwen3_max_token_length": 512,
        "anima_t5_max_token_length": 512,
        "anima_attn_mode": "torch",
        "anima_split_attn": False,
        "anima_vae_chunk_size": 64,
        "anima_vae_disable_cache": True,
        "anima_qwen_image_vae_2d": False,
        "anima_compile": False,
        "anima_torch_compile": False,
        "anima_compile_backend": "inductor",
        "anima_compile_mode": "default",
        "anima_compile_dynamic": "auto",
        "anima_compile_fullgraph": False,
        "anima_compile_cache_size_limit": 0,
        "cond_emb_dim": 32,
        "lllite_mlp_dim": 64,
        "lllite_target_layers": "self_attn_q",
        "lllite_cond_dim": 64,
        "lllite_cond_resblocks": 1,
        "lllite_use_aspp": False,
        "lllite_dropout": 0,
        "lllite_multiplier": 1.0,
        "network_weights": "",
        "lllite_cond_in_channels": 3,
        "lllite_inpaint_masked_input": False,
        "learning_rate": 5e-5,
        "optimizer": "AdamW8bit",
        "optimizer_args": "",
        "lr_scheduler": "constant",
        "lr_scheduler_args": "",
        "lr_warmup_steps": 0,
        "lr_scheduler_num_cycles": "",
        "lr_scheduler_power": 1,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "full_fp16": False,
        "full_bf16": False,
        "mixed_precision": "bf16",
        "num_cpu_threads_per_process": 1,
        "num_processes": 1,
        "num_machines": 1,
        "multi_gpu": False,
        "gpu_ids": "",
        "main_process_port": 0,
        "dynamo_backend": "no",
        "dynamo_mode": "default",
        "dynamo_use_fullgraph": False,
        "dynamo_use_dynamic": False,
        "extra_accelerate_launch_args": "",
        "save_every_n_epochs": 1,
        "save_every_n_steps": 0,
        "save_last_n_steps": 0,
        "save_last_n_steps_state": 0,
        "save_state": False,
        "save_state_on_train_end": False,
        "resume": "",
        "logging_dir": "",
        "log_with": "",
        "log_tracker_name": "",
        "log_tracker_config": "",
        "log_config": False,
        "wandb_api_key": "",
        "wandb_run_name": "",
        "show_timesteps": "",
        "show_timesteps_resolution": "1024",
    }

    missing = [p for p in params if p not in defaults]
    assert not missing, f"_default_kwargs is missing params: {missing}"

    kwargs = {p: defaults[p] for p in params}
    kwargs["headless"] = True
    kwargs["print_only"] = True
    if overrides:
        kwargs.update(overrides)
    return kwargs


def _make_path_fixtures(tmpdir):
    """Create placeholder model files and dataset dirs so path validation passes."""
    models = os.path.join(tmpdir, "models")
    train = os.path.join(tmpdir, "train")
    cond = os.path.join(tmpdir, "cond")
    os.makedirs(models, exist_ok=True)
    os.makedirs(train, exist_ok=True)
    os.makedirs(cond, exist_ok=True)

    dit = os.path.join(models, "anima_dit.safetensors")
    qwen3 = os.path.join(models, "qwen3_0.6b.safetensors")
    vae = os.path.join(models, "qwen_image_vae_fp16.safetensors")
    for path in (dit, qwen3, vae):
        with open(path, "wb") as f:
            f.write(b"0")

    return {
        "pretrained_model_name_or_path": dit,
        "anima_qwen3": qwen3,
        "anima_vae": vae,
        "train_data_dir": train,
        "conditioning_data_dir": cond,
        "output_dir": os.path.join(tmpdir, "out"),
    }


def _run_and_load_toml(overrides=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        path_defaults = _make_path_fixtures(tmpdir)
        os.makedirs(path_defaults["output_dir"], exist_ok=True)
        kwargs = _default_kwargs({**path_defaults, **(overrides or {})})
        # If a test overrides dataset_config only, keep real conditioning
        # paths unless the override explicitly blanks them.
        mock_executor(anima_lllite_gui)
        with patch.object(
            anima_lllite_gui, "get_executable_path", return_value="accelerate"
        ):
            with patch.object(anima_lllite_gui, "print_command_and_toml") as mocked:
                anima_lllite_gui.train_model(**kwargs)
                assert mocked.called, "train_model did not emit a command"
                run_cmd = mocked.call_args[0][0]
                tmpfilename = mocked.call_args[0][1]
                with open(tmpfilename, encoding="utf-8") as f:
                    config = toml.load(f)
        return run_cmd, config, path_defaults


class TestAnimaLlliteCommandGeneration(unittest.TestCase):
    def test_invokes_anima_train_control_net_lllite(self):
        run_cmd, _config, _paths = _run_and_load_toml({})
        self.assertTrue(
            any("anima_train_control_net_lllite.py" in part for part in run_cmd)
        )

    def test_does_not_invoke_lora_or_sdxl_lllite_scripts(self):
        run_cmd, _config, _paths = _run_and_load_toml({})
        joined = " ".join(run_cmd)
        self.assertNotIn("anima_train_network.py", joined)
        self.assertNotIn("sdxl_train_control_net_lllite.py", joined)

    def test_conditioning_dataset_paths_are_forwarded(self):
        _run_cmd, config, paths = _run_and_load_toml({})
        self.assertEqual(config.get("train_data_dir"), paths["train_data_dir"])
        self.assertEqual(
            config.get("conditioning_data_dir"), paths["conditioning_data_dir"]
        )

    def test_dataset_config_skips_folder_args(self):
        _run_cmd, config, _paths = _run_and_load_toml(
            {
                "dataset_config": "test/config/dataset.toml",
                "train_data_dir": "/ignored/train",
                "conditioning_data_dir": "/ignored/cond",
            }
        )
        self.assertEqual(config.get("dataset_config"), "test/config/dataset.toml")
        self.assertNotIn("train_data_dir", config)
        self.assertNotIn("conditioning_data_dir", config)

    def test_missing_dataset_aborts_training(self):
        mock_executor(anima_lllite_gui)
        with patch.object(anima_lllite_gui, "print_command_and_toml") as mocked:
            with patch.object(
                anima_lllite_gui, "get_executable_path", return_value="accelerate"
            ):
                kwargs = _default_kwargs(
                    {
                        "train_data_dir": "",
                        "conditioning_data_dir": "",
                        "dataset_config": "",
                    }
                )
                anima_lllite_gui.train_model(**kwargs)
                mocked.assert_not_called()

    def test_missing_conditioning_dir_aborts_without_dataset_config(self):
        mock_executor(anima_lllite_gui)
        with tempfile.TemporaryDirectory() as tmpdir:
            train = os.path.join(tmpdir, "train")
            os.makedirs(train)
            with patch.object(anima_lllite_gui, "print_command_and_toml") as mocked:
                with patch.object(
                    anima_lllite_gui, "get_executable_path", return_value="accelerate"
                ):
                    kwargs = _default_kwargs(
                        {
                            "train_data_dir": train,
                            "conditioning_data_dir": "",
                            "dataset_config": "",
                        }
                    )
                    anima_lllite_gui.train_model(**kwargs)
                    mocked.assert_not_called()

    def test_lllite_target_layers_are_forwarded(self):
        _run_cmd, config, _paths = _run_and_load_toml(
            {"lllite_target_layers": "self_attn_q_pre,mlp_fc1_pre"}
        )
        self.assertEqual(
            config.get("lllite_target_layers"), "self_attn_q_pre,mlp_fc1_pre"
        )

    def test_lllite_use_aspp_forwarded_when_enabled(self):
        _run_cmd, config, _paths = _run_and_load_toml({"lllite_use_aspp": True})
        self.assertTrue(config.get("lllite_use_aspp"))

    def test_lllite_use_aspp_omitted_when_disabled(self):
        _run_cmd, config, _paths = _run_and_load_toml({"lllite_use_aspp": False})
        self.assertNotIn("lllite_use_aspp", config)

    def test_lllite_dims_are_forwarded(self):
        _run_cmd, config, _paths = _run_and_load_toml(
            {
                "cond_emb_dim": 48,
                "lllite_mlp_dim": 96,
                "lllite_cond_dim": 80,
                "lllite_cond_resblocks": 2,
                "lllite_multiplier": 0.5,
            }
        )
        self.assertEqual(config.get("cond_emb_dim"), 48)
        self.assertEqual(config.get("lllite_mlp_dim"), 96)
        self.assertEqual(config.get("lllite_cond_dim"), 80)
        self.assertEqual(config.get("lllite_cond_resblocks"), 2)
        self.assertEqual(config.get("lllite_multiplier"), 0.5)

    def test_inpaint_cond_channels_and_mask_flag(self):
        _run_cmd, config, _paths = _run_and_load_toml(
            {
                "lllite_cond_in_channels": 4,
                "lllite_inpaint_masked_input": True,
            }
        )
        self.assertEqual(config.get("lllite_cond_in_channels"), 4)
        self.assertTrue(config.get("lllite_inpaint_masked_input"))

    def test_anima_model_paths_are_forwarded(self):
        _run_cmd, config, paths = _run_and_load_toml({})
        self.assertEqual(config.get("qwen3"), paths["anima_qwen3"])
        self.assertEqual(config.get("vae"), paths["anima_vae"])
        self.assertEqual(
            config.get("pretrained_model_name_or_path"),
            paths["pretrained_model_name_or_path"],
        )

    def test_optional_anima_paths_omitted_when_blank(self):
        _run_cmd, config, _paths = _run_and_load_toml({})
        self.assertNotIn("llm_adapter_path", config)
        self.assertNotIn("t5_tokenizer_path", config)

    def test_flow_matching_params_are_forwarded(self):
        _run_cmd, config, _paths = _run_and_load_toml({})
        self.assertEqual(config.get("discrete_flow_shift"), 3.0)
        self.assertEqual(config.get("timestep_sampling"), "shift")
        self.assertEqual(config.get("sigmoid_scale"), 1.0)

    def test_unsupported_mvp_flags_are_never_emitted(self):
        """Backend asserts if these are set; the GUI must not write them."""
        _run_cmd, config, _paths = _run_and_load_toml({})
        for key in (
            "blocks_to_swap",
            "cpu_offload_checkpointing",
            "unsloth_offload_checkpointing",
            "deepspeed",
            "fused_backward_pass",
        ):
            self.assertNotIn(key, config)

    def test_compile_and_torch_compile_together_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_defaults = _make_path_fixtures(tmpdir)
            mock_executor(anima_lllite_gui)
            with patch.object(anima_lllite_gui, "print_command_and_toml") as mocked:
                with patch.object(
                    anima_lllite_gui, "get_executable_path", return_value="accelerate"
                ):
                    kwargs = _default_kwargs(
                        {
                            **path_defaults,
                            "anima_compile": True,
                            "anima_torch_compile": True,
                        }
                    )
                    anima_lllite_gui.train_model(**kwargs)
                    mocked.assert_not_called()

    def test_missing_anima_model_paths_abort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_defaults = _make_path_fixtures(tmpdir)
            mock_executor(anima_lllite_gui)
            with patch.object(anima_lllite_gui, "print_command_and_toml") as mocked:
                with patch.object(
                    anima_lllite_gui, "get_executable_path", return_value="accelerate"
                ):
                    kwargs = _default_kwargs(
                        {
                            **path_defaults,
                            "anima_qwen3": "",
                            "anima_vae": "",
                        }
                    )
                    anima_lllite_gui.train_model(**kwargs)
                    mocked.assert_not_called()


class TestAnimaLlliteFieldRegistry(unittest.TestCase):
    """FIELD_REGISTRY must stay aligned with train_model/save/open signatures."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            anima_lllite_gui.anima_lllite_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = anima_lllite_gui.last_built_field_registry

    def test_field_registry_was_built(self):
        self.assertIsNotNone(self.field_registry)
        self.assertGreater(len(self.field_registry), 0)

    def test_field_registry_names_are_unique(self):
        names = [name for name, _ in self.field_registry]
        self.assertEqual(len(names), len(set(names)))

    def test_field_registry_matches_train_model_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        train_model_params = list(
            inspect.signature(anima_lllite_gui.train_model).parameters
        )
        self.assertEqual(registry_names, train_model_params[2:])

    def test_field_registry_matches_save_configuration_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        save_config_params = list(
            inspect.signature(anima_lllite_gui.save_configuration).parameters
        )
        # save_configuration's first two params are save_as_bool, file_path
        self.assertEqual(registry_names, save_config_params[2:])

    def test_field_registry_matches_open_configuration_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        open_config_params = list(
            inspect.signature(anima_lllite_gui.open_configuration).parameters
        )
        # open_configuration's first two params are ask_for_file, file_path
        self.assertEqual(registry_names, open_config_params[2:])

    def test_unsupported_mvp_widgets_are_not_in_registry(self):
        names = {name for name, _ in self.field_registry}
        for forbidden in (
            "blocks_to_swap",
            "cpu_offload_checkpointing",
            "unsloth_offload_checkpointing",
            "deepspeed",
            "fused_backward_pass",
        ):
            self.assertNotIn(forbidden, names)


if __name__ == "__main__":
    unittest.main()
