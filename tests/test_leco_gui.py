"""Regression tests for GH issue #3526: LECO training support in the GUI."""

import inspect
import os
import tempfile
import unittest
from unittest.mock import patch

import toml

from kohya_gui import leco_gui
from conftest import mock_executor


def _default_kwargs(overrides=None):
    sig = inspect.signature(leco_gui.train_model)
    params = [p for p in sig.parameters if p not in ("headless", "print_only")]

    defaults = {
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "v2": False,
        "v_parameterization": False,
        "sdxl": False,
        "output_dir": "",
        "output_name": "test_leco",
        "save_model_as": "safetensors",
        "save_precision": "fp16",
        "training_comment": "",
        "no_metadata": False,
        "prompts_file": "test/config/leco_prompts.toml",
        "network_module": "networks.lora",
        "network_dim": 8,
        "network_alpha": 4,
        "network_dropout": 0,
        "network_args": "",
        "network_weights": "",
        "dim_from_weights": False,
        "learning_rate": 0.0001,
        "unet_lr": 0.0001,
        "optimizer": "AdamW8bit",
        "optimizer_args": "",
        "lr_scheduler": "constant",
        "lr_scheduler_args": "",
        "lr_warmup_steps": 0,
        "lr_scheduler_num_cycles": "",
        "lr_scheduler_power": 1,
        "max_train_steps": 500,
        "max_grad_norm": 1.0,
        "max_denoising_steps": 40,
        "leco_denoise_guidance_scale": 3.0,
        "seed": 0,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "num_cpu_threads_per_process": 2,
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
        "gradient_checkpointing": True,
        "full_fp16": False,
        "full_bf16": False,
        "xformers": "sdpa",
        "mem_eff_attn": False,
        "clip_skip": 1,
        "noise_offset": 0,
        "zero_terminal_snr": False,
        "min_snr_gamma": 0,
        "save_every_n_steps": 100,
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
    }

    missing = [p for p in params if p not in defaults]
    assert not missing, f"_default_kwargs is missing params: {missing}"

    kwargs = {p: defaults[p] for p in params}
    kwargs["headless"] = True
    kwargs["print_only"] = True
    if overrides:
        kwargs.update(overrides)
    return kwargs


def _run_and_load_toml(overrides=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        kwargs = _default_kwargs({**(overrides or {}), "output_dir": tmpdir})
        mock_executor(leco_gui)
        with patch.object(leco_gui, "print_command_and_toml") as mocked:
            leco_gui.train_model(**kwargs)
            run_cmd = mocked.call_args[0][0]
            tmpfilename = mocked.call_args[0][1]
            with open(tmpfilename, encoding="utf-8") as f:
                config = toml.load(f)
        return run_cmd, config


class TestLecoGuiCommandGeneration(unittest.TestCase):
    def test_sd_backend_invokes_train_leco(self):
        run_cmd, config = _run_and_load_toml({"sdxl": False})
        self.assertTrue(any("train_leco.py" in part for part in run_cmd))
        self.assertFalse(any("sdxl_train_leco.py" in part for part in run_cmd))

    def test_sdxl_backend_invokes_sdxl_train_leco(self):
        run_cmd, config = _run_and_load_toml({"sdxl": True})
        self.assertTrue(any("sdxl_train_leco.py" in part for part in run_cmd))

    def test_prompts_file_is_required_argument(self):
        run_cmd, config = _run_and_load_toml({})
        self.assertEqual(config.get("prompts_file"), "test/config/leco_prompts.toml")

    def test_missing_prompts_file_aborts_training(self):
        mock_executor(leco_gui)
        with patch.object(leco_gui, "print_command_and_toml") as mocked:
            kwargs = _default_kwargs({"prompts_file": ""})
            leco_gui.train_model(**kwargs)
            mocked.assert_not_called()

    def test_no_dataset_folder_args_in_config(self):
        run_cmd, config = _run_and_load_toml({})
        self.assertNotIn("train_data_dir", config)
        self.assertNotIn("dataset_config", config)
        self.assertNotIn("max_train_epochs", config)

    def test_network_and_leco_specific_params_are_forwarded(self):
        run_cmd, config = _run_and_load_toml(
            {
                "network_dim": 16,
                "max_denoising_steps": 30,
                "leco_denoise_guidance_scale": 2.5,
            }
        )
        self.assertEqual(config.get("network_dim"), 16)
        self.assertEqual(config.get("max_denoising_steps"), 30)
        self.assertEqual(config.get("leco_denoise_guidance_scale"), 2.5)

    def test_save_every_n_steps_is_forwarded(self):
        run_cmd, config = _run_and_load_toml({"save_every_n_steps": 200})
        self.assertEqual(config.get("save_every_n_steps"), 200)


if __name__ == "__main__":
    unittest.main()
