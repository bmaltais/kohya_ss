"""Regression tests for GH issue #3520: kohya_gui emitting args that
sd-scripts v0.11.1 silently ignores (LoRA+ ratios, stale `lowvram`).

Also covers GH issue #3543 Milestone 1: `FIELD_REGISTRY` must stay in
identical relative order with train_model's/save_configuration's/
open_configuration's shared keyword-argument order (see
TestLoraGuiFieldRegistry below), and Milestone 2: the train/save/load/
preset buttons' actual `.click()`/`.input()` callables must look up each
argument by component identity rather than position (see
TestLoraGuiDictAdapterWiring below).
"""

import inspect
import json
import os
import tempfile
import unittest

import gradio as gr
import toml

from kohya_gui import lora_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from conftest import (
    build_train_model_kwargs,
    mock_executor,
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


class TestLoraGuiFieldRegistry(unittest.TestCase):
    """GH issue #3543 Milestone 1: `FIELD_REGISTRY` (the source `settings_list`
    is now derived from) must declare its field names in the exact same
    relative order as train_model's/save_configuration's/open_configuration's
    shared keyword-argument order, since a positional-order mismatch there
    silently shifts every subsequent value into the wrong parameter.
    """

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = lora_gui.last_built_field_registry

    def test_field_registry_was_built(self):
        self.assertIsNotNone(self.field_registry)
        self.assertGreater(len(self.field_registry), 0)

    def test_field_registry_names_are_unique(self):
        names = [name for name, _ in self.field_registry]
        self.assertEqual(len(names), len(set(names)))

    def test_field_registry_matches_train_model_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        train_model_params = list(inspect.signature(lora_gui.train_model).parameters)
        # train_model's first two params (headless, print_only) are supplied
        # separately by the .click() wiring, not via settings_list.
        self.assertEqual(registry_names, train_model_params[2:])

    def test_field_registry_matches_save_configuration_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        save_config_params = list(
            inspect.signature(lora_gui.save_configuration).parameters
        )
        # save_configuration's first two params (save_as_bool, file_path) are
        # supplied separately, not via settings_list.
        self.assertEqual(registry_names, save_config_params[2:])

    def test_field_registry_matches_open_configuration_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        open_config_params = list(
            inspect.signature(lora_gui.open_configuration).parameters
        )
        # open_configuration's first three params (ask_for_file, apply_preset,
        # file_path) and trailing training_preset are supplied separately.
        self.assertEqual(registry_names, open_config_params[3:-1])


class TestLoraGuiDictAdapterWiring(unittest.TestCase):
    """GH issue #3543 Milestone 2: the train/save/load/preset buttons' real
    `.click()`/`.input()` callables must resolve each argument by component
    identity (a `dict[Component, value]`, keyed off FIELD_REGISTRY) rather
    than by position. These tests call the actual bound callables exposed via
    `last_built_gui_entries` -- the same objects Gradio invokes -- instead of
    calling train_model/save_configuration/open_configuration directly, so
    they exercise the wiring itself, not just the underlying functions.
    """

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = lora_gui.last_built_field_registry
        cls.entries = lora_gui.last_built_gui_entries
        cls.components = cls.entries["components"]
        cls.settings_list = [comp for _, comp in cls.field_registry]

    def _field_data(self, kwargs: dict) -> dict:
        return {comp: kwargs[name] for name, comp in self.field_registry}

    def _field_kwargs(self, overrides=None):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=overrides,
        )
        kwargs.pop("headless")
        kwargs.pop("print_only")
        return kwargs

    def test_train_model_entry_missing_component_raises_keyerror(self):
        # Proves the wiring fails loudly (KeyError) rather than silently
        # shifting values, the core hazard #3543 was filed to eliminate.
        # `_field_data` never populates the dummy headless/print_only
        # components, so calling the entry without adding them back exercises
        # exactly that failure mode.
        kwargs = self._field_kwargs()
        data = self._field_data(kwargs)
        with self.assertRaises(KeyError):
            self.entries["train_model"](data)

    # Keys whose value embeds the scratch output_dir path, so a fresh tmpdir
    # per call makes them differ trivially -- not what this test verifies.
    PATH_DEPENDENT_KEYS = ("output_dir", "sample_prompts")

    def test_print_command_entry_matches_direct_train_model_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._field_kwargs(
                {"output_dir": tmpdir, "output_name": "wiring_test"}
            )
            data = self._field_data(kwargs)
            data[self.components["dummy_headless"]] = True
            data[self.components["dummy_db_true"]] = True

            mock_executor(lora_gui)
            self.entries["print_command"](data)
            toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
            self.assertEqual(len(toml_files), 1)
            with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                via_wiring = toml.load(f)

        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._field_kwargs(
                {"output_dir": tmpdir, "output_name": "wiring_test"}
            )
            mock_executor(lora_gui)
            lora_gui.train_model(headless=True, print_only=True, **kwargs)
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

        # Close the handle before reopening: Windows locks NamedTemporaryFile
        # while the with-block holds it open, which fails save_configuration/open.
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
            lora_gui.save_configuration(
                save_as_bool=False, file_path=direct_path, **kwargs
            )
            with open(direct_path, encoding="utf-8") as f:
                via_direct_call = json.load(f)
        finally:
            os.unlink(direct_path)

        self.assertEqual(via_wiring, via_direct_call)

    def test_load_configuration_entry_matches_direct_open_configuration_call(self):
        kwargs = self._field_kwargs()
        data = self._field_data(kwargs)
        data[self.components["dummy_db_false"]] = False
        data[self.components["config_file_name"]] = FIXTURE
        data[self.components["training_preset"]] = "none"

        result = self.entries["load_configuration"](data)

        direct_result = lora_gui.open_configuration(
            ask_for_file=False,
            apply_preset=False,
            file_path=FIXTURE,
            training_preset="none",
            **kwargs,
        )

        # direct_result is [file_path, *settings_list-ordered values,
        # training_preset, convolution_row]; compare everything but the
        # trailing gr.Row visibility object (not equality-comparable).
        self.assertEqual(result[self.components["config_file_name"]], direct_result[0])
        for comp, expected in zip(self.settings_list, direct_result[1:-2]):
            self.assertEqual(result[comp], expected)
        self.assertEqual(result[self.components["training_preset"]], direct_result[-2])


if __name__ == "__main__":
    unittest.main()
