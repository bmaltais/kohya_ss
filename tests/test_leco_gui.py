"""Regression tests for GH issue #3526: LECO training support in the GUI.

Also covers GH issue #3543 Milestone 3 (leco_gui.py): `FIELD_REGISTRY` must
stay in identical relative order with train_model's/save_configuration's/
open_configuration's shared keyword-argument order, and the train/save/load
buttons' actual `.click()` callables must look up each argument by component
identity rather than position.
"""

import inspect
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
import toml

from kohya_gui import leco_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig
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
        # train_model aborts if accelerate is missing from PATH; pin a
        # fake path so the suite doesn't depend on the host environment.
        with patch.object(leco_gui, "get_executable_path", return_value="accelerate"):
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


class TestLecoGuiFieldRegistry(unittest.TestCase):
    """GH issue #3543 M3: `FIELD_REGISTRY` (the source `settings_list` is now
    derived from) must declare its field names in the exact same relative order
    as train_model's/save_configuration's/open_configuration's shared
    keyword-argument order.
    """

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            leco_gui.leco_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = leco_gui.last_built_field_registry

    def test_field_registry_was_built(self):
        self.assertIsNotNone(self.field_registry)
        self.assertGreater(len(self.field_registry), 0)

    def test_field_registry_names_are_unique(self):
        names = [name for name, _ in self.field_registry]
        self.assertEqual(len(names), len(set(names)))

    def test_field_registry_matches_train_model_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        train_model_params = list(inspect.signature(leco_gui.train_model).parameters)
        # train_model's first two params (headless, print_only) are supplied
        # separately by the .click() wiring, not via settings_list.
        self.assertEqual(registry_names, train_model_params[2:])

    def test_field_registry_matches_save_configuration_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        save_config_params = list(
            inspect.signature(leco_gui.save_configuration).parameters
        )
        # save_configuration's first two params (save_as_bool, file_path) are
        # supplied separately, not via settings_list.
        self.assertEqual(registry_names, save_config_params[2:])

    def test_field_registry_matches_open_configuration_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        open_config_params = list(
            inspect.signature(leco_gui.open_configuration).parameters
        )
        # open_configuration's first two params (ask_for_file, file_path) are
        # supplied separately; there is no trailing training_preset on LECO.
        self.assertEqual(registry_names, open_config_params[2:])


class TestLecoGuiDictAdapterWiring(unittest.TestCase):
    """GH issue #3543 M3: the train/save/load buttons' real `.click()` callables
    must resolve each argument by component identity (a `dict[Component, value]`,
    keyed off FIELD_REGISTRY) rather than by position.
    """

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            leco_gui.leco_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = leco_gui.last_built_field_registry
        cls.entries = leco_gui.last_built_gui_entries
        cls.components = cls.entries["components"]
        cls.settings_list = [comp for _, comp in cls.field_registry]

    def _field_data(self, kwargs: dict) -> dict:
        return {comp: kwargs[name] for name, comp in self.field_registry}

    def _field_kwargs(self, overrides=None):
        kwargs = _default_kwargs(overrides)
        kwargs.pop("headless")
        kwargs.pop("print_only")
        return kwargs

    def test_train_model_entry_missing_component_raises_keyerror(self):
        # Proves the wiring fails loudly (KeyError) rather than silently
        # shifting values — the core hazard #3543 was filed to eliminate.
        kwargs = self._field_kwargs()
        data = self._field_data(kwargs)
        with self.assertRaises(KeyError):
            self.entries["train_model"](data)

    PATH_DEPENDENT_KEYS = ("output_dir",)

    def test_print_command_entry_matches_direct_train_model_call(self):
        with patch.object(leco_gui, "get_executable_path", return_value="accelerate"):
            with tempfile.TemporaryDirectory() as tmpdir:
                kwargs = self._field_kwargs(
                    {"output_dir": tmpdir, "output_name": "wiring_test"}
                )
                data = self._field_data(kwargs)
                data[self.components["dummy_headless"]] = True
                data[self.components["dummy_db_true"]] = True

                mock_executor(leco_gui)
                self.entries["print_command"](data)
                toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
                self.assertEqual(len(toml_files), 1)
                with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                    via_wiring = toml.load(f)

            with tempfile.TemporaryDirectory() as tmpdir:
                kwargs = self._field_kwargs(
                    {"output_dir": tmpdir, "output_name": "wiring_test"}
                )
                mock_executor(leco_gui)
                leco_gui.train_model(headless=True, print_only=True, **kwargs)
                toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
                with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                    via_direct_call = toml.load(f)

        for key in self.PATH_DEPENDENT_KEYS:
            via_wiring.pop(key, None)
            via_direct_call.pop(key, None)
        self.assertEqual(via_wiring, via_direct_call)

    def test_save_configuration_entry_matches_direct_call(self):
        kwargs = self._field_kwargs()
        data = self._field_data(kwargs)

        via_wiring_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        via_wiring_path = via_wiring_file.name
        via_wiring_file.close()
        try:
            data[self.components["dummy_db_false"]] = False
            data[self.components["config_file_name"]] = via_wiring_path
            self.entries["save_configuration"](data)
            with open(via_wiring_path, encoding="utf-8") as f:
                via_wiring = json.load(f)
        finally:
            os.unlink(via_wiring_path)

        direct_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        direct_path = direct_file.name
        direct_file.close()
        try:
            leco_gui.save_configuration(
                save_as_bool=False, file_path=direct_path, **kwargs
            )
            with open(direct_path, encoding="utf-8") as f:
                via_direct_call = json.load(f)
        finally:
            os.unlink(direct_path)

        self.assertEqual(via_wiring, via_direct_call)

    def test_load_configuration_entry_matches_direct_open_configuration_call(self):
        # Round-trip a known config through save, then load via the adapter
        # so we exercise the real component-keyed path without needing a
        # checked-in LECO fixture JSON.
        kwargs = self._field_kwargs()
        config_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        config_path = config_file.name
        config_file.close()
        try:
            leco_gui.save_configuration(
                save_as_bool=False, file_path=config_path, **kwargs
            )

            data = self._field_data(kwargs)
            data[self.components["dummy_db_false"]] = False
            data[self.components["config_file_name"]] = config_path

            result = self.entries["load_configuration"](data)

            direct_result = leco_gui.open_configuration(
                ask_for_file=False,
                file_path=config_path,
                **kwargs,
            )

            self.assertEqual(
                result[self.components["config_file_name"]], direct_result[0]
            )
            for comp, expected in zip(self.settings_list, direct_result[1:]):
                self.assertEqual(result[comp], expected)
        finally:
            os.unlink(config_path)

    def test_open_configuration_entry_requests_file_dialog(self):
        # Open must pass ask_for_file=True (unlike Load). The pre-adapter
        # wiring used False for both; assert the fixed distinction holds.
        kwargs = self._field_kwargs()
        data = self._field_data(kwargs)
        data[self.components["dummy_db_true"]] = True
        data[self.components["config_file_name"]] = ""

        with patch.object(leco_gui, "get_file_path", return_value="") as mocked_picker:
            # Empty picker result falls through to original_file_path and
            # returns a full tuple of current values — no KeyError/None.
            result = self.entries["open_configuration"](data)
            mocked_picker.assert_called_once()
            self.assertEqual(result[self.components["config_file_name"]], "")


if __name__ == "__main__":
    unittest.main()
