"""Regression test for GH issue #3520: dreambooth_gui.py leaking LoRA-only
`split_mode`/`train_blocks` into the flux_train.py (full fine-tune) config,
the same bug fixed in finetune_gui.py.

Also covers GH issue #3527: inpainting model training support (SD1.5/SDXL).
`--train_inpainting` must be forwarded to train_db.py/sdxl_train.py and must
never be combined with `--cache_latents`/`--cache_latents_to_disk` since
masks are generated randomly per step from the source image. It must also
never leak into flux_train.py/sd3_train.py, which aren't in the supported
script list.

Also covers GH issue #3543 Milestone 3: FIELD_REGISTRY + dict-keyed adapters.
"""

import inspect
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr

from kohya_gui import dreambooth_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from conftest import (
    build_train_model_kwargs,
    mock_executor,
    run_train_model_and_load_saved_json,
    run_train_model_and_load_toml,
)

FIXTURE = "test/config/dreambooth-AdamW.json"
NUMERIC_FIXUPS = ("max_grad_norm",)


class TestDreamboothFluxConfigOutput(unittest.TestCase):
    def test_split_mode_and_train_blocks_never_reach_flux_train_config(self):
        kwargs = build_train_model_kwargs(
            dreambooth_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            overrides={
                "flux1_checkbox": True,
                "split_mode": True,
                "train_blocks": "all",
            },
        )
        config = run_train_model_and_load_toml(dreambooth_gui, kwargs)

        self.assertNotIn("split_mode", config)
        self.assertNotIn("train_blocks", config)


class TestDreamboothTrainInpainting(unittest.TestCase):
    def test_train_inpainting_forwarded_and_cache_latents_dropped(self):
        kwargs = build_train_model_kwargs(
            dreambooth_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            overrides={
                "train_inpainting": True,
                "cache_latents": True,
                "cache_latents_to_disk": True,
            },
        )
        config = run_train_model_and_load_toml(dreambooth_gui, kwargs)

        self.assertTrue(config.get("train_inpainting"))
        self.assertNotIn("cache_latents", config)
        self.assertNotIn("cache_latents_to_disk", config)

    def test_train_inpainting_off_leaves_cache_latents_untouched(self):
        kwargs = build_train_model_kwargs(
            dreambooth_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            overrides={
                "train_inpainting": False,
                "cache_latents": True,
                "cache_latents_to_disk": False,
            },
        )
        config = run_train_model_and_load_toml(dreambooth_gui, kwargs)

        self.assertNotIn("train_inpainting", config)
        self.assertTrue(config.get("cache_latents"))

    def test_train_inpainting_dropped_for_flux_backend(self):
        # train_inpainting is only supported by train_db.py/sdxl_train.py;
        # flux_train.py must never receive it.
        kwargs = build_train_model_kwargs(
            dreambooth_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            overrides={
                "train_inpainting": True,
                "flux1_checkbox": True,
            },
        )
        config = run_train_model_and_load_toml(dreambooth_gui, kwargs)

        self.assertNotIn("train_inpainting", config)

    def test_saved_json_config_reflects_forced_overrides(self):
        # The JSON training config saved via SaveConfigFile() must not
        # persist the pre-override checkbox state (train_inpainting=True
        # with cache_latents=True), which would produce an invalid combo
        # if the user reloads this saved config later.
        kwargs = build_train_model_kwargs(
            dreambooth_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            overrides={
                "train_inpainting": True,
                "cache_latents": True,
                "cache_latents_to_disk": True,
            },
        )
        config = run_train_model_and_load_saved_json(dreambooth_gui, kwargs)

        self.assertTrue(config.get("train_inpainting"))
        self.assertFalse(config.get("cache_latents"))
        self.assertFalse(config.get("cache_latents_to_disk"))


class TestDreamboothGuiFieldRegistry(unittest.TestCase):
    """GH #3543 M3: FIELD_REGISTRY must match train/save/open keyword order."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            dreambooth_gui.dreambooth_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = dreambooth_gui.last_built_field_registry

    def test_field_registry_was_built(self):
        self.assertIsNotNone(self.field_registry)
        self.assertGreater(len(self.field_registry), 0)

    def test_field_registry_names_are_unique(self):
        names = [name for name, _ in self.field_registry]
        self.assertEqual(len(names), len(set(names)))

    def test_field_registry_matches_train_model_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        train_model_params = list(
            inspect.signature(dreambooth_gui.train_model).parameters
        )
        self.assertEqual(registry_names, train_model_params[2:])

    def test_field_registry_matches_save_configuration_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        save_config_params = list(
            inspect.signature(dreambooth_gui.save_configuration).parameters
        )
        self.assertEqual(registry_names, save_config_params[2:])

    def test_field_registry_matches_open_configuration_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        open_config_params = list(
            inspect.signature(dreambooth_gui.open_configuration).parameters
        )
        self.assertEqual(registry_names, open_config_params[2:])


class TestDreamboothGuiDictAdapterWiring(unittest.TestCase):
    """GH #3543 M3: train/save/load buttons resolve args by component identity."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            dreambooth_gui.dreambooth_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = dreambooth_gui.last_built_field_registry
        cls.entries = dreambooth_gui.last_built_gui_entries
        cls.components = cls.entries["components"]
        cls.settings_list = [comp for _, comp in cls.field_registry]

    def _field_data(self, kwargs: dict) -> dict:
        return {comp: kwargs[name] for name, comp in self.field_registry}

    def _field_kwargs(self, overrides=None):
        kwargs = build_train_model_kwargs(
            dreambooth_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            overrides=overrides,
        )
        kwargs.pop("headless")
        kwargs.pop("print_only")
        return kwargs

    def test_train_model_entry_missing_component_raises_keyerror(self):
        kwargs = self._field_kwargs()
        data = self._field_data(kwargs)
        with self.assertRaises(KeyError):
            self.entries["train_model"](data)

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
            dreambooth_gui.save_configuration(
                save_as_bool=False, file_path=direct_path, **kwargs
            )
            with open(direct_path, encoding="utf-8") as f:
                via_direct_call = json.load(f)
        finally:
            os.unlink(direct_path)

        self.assertEqual(via_wiring, via_direct_call)

    def test_load_configuration_entry_matches_direct_open_configuration_call(self):
        kwargs = self._field_kwargs()
        config_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        config_path = config_file.name
        config_file.close()
        try:
            dreambooth_gui.save_configuration(
                save_as_bool=False, file_path=config_path, **kwargs
            )
            data = self._field_data(kwargs)
            data[self.components["dummy_db_false"]] = False
            data[self.components["config_file_name"]] = config_path

            result = self.entries["load_configuration"](data)
            direct_result = dreambooth_gui.open_configuration(
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

    def test_print_command_entry_matches_direct_train_model_call(self):
        import toml

        with patch.object(
            dreambooth_gui, "get_executable_path", return_value="accelerate"
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                kwargs = self._field_kwargs(
                    {"output_dir": tmpdir, "output_name": "wiring_test"}
                )
                data = self._field_data(kwargs)
                data[self.components["dummy_headless"]] = True
                data[self.components["dummy_db_true"]] = True
                mock_executor(dreambooth_gui)
                self.entries["print_command"](data)
                toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
                self.assertEqual(len(toml_files), 1)
                with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                    via_wiring = toml.load(f)

            with tempfile.TemporaryDirectory() as tmpdir:
                kwargs = self._field_kwargs(
                    {"output_dir": tmpdir, "output_name": "wiring_test"}
                )
                mock_executor(dreambooth_gui)
                dreambooth_gui.train_model(headless=True, print_only=True, **kwargs)
                toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
                with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                    via_direct_call = toml.load(f)

        for key in ("output_dir", "sample_prompts"):
            via_wiring.pop(key, None)
            via_direct_call.pop(key, None)
        self.assertEqual(via_wiring, via_direct_call)


if __name__ == "__main__":
    unittest.main()
