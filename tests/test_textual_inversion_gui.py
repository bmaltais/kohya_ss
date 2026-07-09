"""Regression test for GH issue #3520: textual_inversion_gui.py emitting
`stop_text_encoder_training`, an arg only defined by train_db.py — neither
train_textual_inversion.py nor sdxl_train_textual_inversion.py accept it.

Also covers GH issue #3543 Milestone 3: FIELD_REGISTRY + dict-keyed adapters.
"""

import inspect
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
import toml

from kohya_gui import textual_inversion_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from conftest import (
    build_train_model_kwargs,
    mock_executor,
    run_train_model_and_load_toml,
)

FIXTURE = "test/config/TI-AdamW8bit-toml.json"

NUMERIC_FIXUPS = (
    "lr_warmup_steps",
    "stop_text_encoder_training_pct",
    "main_process_port",
    "ip_noise_gamma",
    "ip_noise_gamma_random_strength",
    "huber_c",
    "huber_scale",
    "save_last_n_epochs",
    "save_last_n_epochs_state",
    "noise_offset_random_strength",
    "max_train_epochs",
)
STRING_OVERRIDES = (
    "log_with",
    "log_config",
    "lr_scheduler_type",
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
    "dynamo_backend",
    "dynamo_mode",
    "extra_accelerate_launch_args",
)


class TestTextualInversionConfigOutput(unittest.TestCase):
    def test_stop_text_encoder_training_never_reaches_ti_config(self):
        kwargs = build_train_model_kwargs(
            textual_inversion_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={"stop_text_encoder_training_pct": 50, "max_train_steps": 100},
        )
        config = run_train_model_and_load_toml(textual_inversion_gui, kwargs)

        self.assertNotIn("stop_text_encoder_training", config)


# train_model uses stop_text_encoder_training_pct; save/open keep the
# historical stop_text_encoder_training config key.
_TI_TRAIN_TO_CONFIG = {
    "stop_text_encoder_training_pct": "stop_text_encoder_training",
}


class TestTextualInversionGuiFieldRegistry(unittest.TestCase):
    """GH #3543 M3: FIELD_REGISTRY must match train_model keyword order."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            textual_inversion_gui.ti_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = textual_inversion_gui.last_built_field_registry

    def test_field_registry_was_built(self):
        self.assertIsNotNone(self.field_registry)
        self.assertGreater(len(self.field_registry), 0)

    def test_field_registry_names_are_unique(self):
        names = [name for name, _ in self.field_registry]
        self.assertEqual(len(names), len(set(names)))

    def test_field_registry_matches_train_model_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        train_model_params = list(
            inspect.signature(textual_inversion_gui.train_model).parameters
        )
        self.assertEqual(registry_names, train_model_params[2:])

    def test_field_registry_matches_save_configuration_after_aliases(self):
        registry_names = [
            _TI_TRAIN_TO_CONFIG.get(name, name) for name, _ in self.field_registry
        ]
        save_config_params = list(
            inspect.signature(textual_inversion_gui.save_configuration).parameters
        )
        self.assertEqual(registry_names, save_config_params[2:])

    def test_field_registry_matches_open_configuration_after_aliases(self):
        registry_names = [
            _TI_TRAIN_TO_CONFIG.get(name, name) for name, _ in self.field_registry
        ]
        open_config_params = list(
            inspect.signature(textual_inversion_gui.open_configuration).parameters
        )
        self.assertEqual(registry_names, open_config_params[2:])


class TestTextualInversionGuiDictAdapterWiring(unittest.TestCase):
    """GH #3543 M3: train/save/load buttons resolve args by component identity."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            textual_inversion_gui.ti_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = textual_inversion_gui.last_built_field_registry
        cls.entries = textual_inversion_gui.last_built_gui_entries
        cls.components = cls.entries["components"]
        cls.settings_list = [comp for _, comp in cls.field_registry]

    def _field_data(self, kwargs: dict) -> dict:
        return {comp: kwargs[name] for name, comp in self.field_registry}

    def _field_kwargs(self, overrides=None):
        kwargs = build_train_model_kwargs(
            textual_inversion_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=overrides,
        )
        kwargs.pop("headless")
        kwargs.pop("print_only")
        return kwargs

    def _config_kwargs(self, kwargs: dict) -> dict:
        return {_TI_TRAIN_TO_CONFIG.get(k, k): v for k, v in kwargs.items()}

    def test_train_model_entry_missing_component_raises_keyerror(self):
        kwargs = self._field_kwargs()
        data = self._field_data(kwargs)
        with self.assertRaises(KeyError):
            self.entries["train_model"](data)

    def test_save_configuration_entry_matches_direct_call(self):
        kwargs = self._field_kwargs()
        data = self._field_data(kwargs)
        config_kwargs = self._config_kwargs(kwargs)

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
            textual_inversion_gui.save_configuration(
                save_as_bool=False, file_path=direct_path, **config_kwargs
            )
            with open(direct_path, encoding="utf-8") as f:
                via_direct_call = json.load(f)
        finally:
            os.unlink(direct_path)

        self.assertEqual(via_wiring, via_direct_call)

    def test_load_configuration_entry_matches_direct_open_configuration_call(self):
        kwargs = self._field_kwargs()
        config_kwargs = self._config_kwargs(kwargs)
        config_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        config_path = config_file.name
        config_file.close()
        try:
            textual_inversion_gui.save_configuration(
                save_as_bool=False, file_path=config_path, **config_kwargs
            )
            data = self._field_data(kwargs)
            data[self.components["dummy_db_false"]] = False
            data[self.components["config_file_name"]] = config_path

            result = self.entries["load_configuration"](data)
            direct_result = textual_inversion_gui.open_configuration(
                ask_for_file=False,
                file_path=config_path,
                **config_kwargs,
            )

            self.assertEqual(
                result[self.components["config_file_name"]], direct_result[0]
            )
            for comp, expected in zip(self.settings_list, direct_result[1:]):
                self.assertEqual(result[comp], expected)
        finally:
            os.unlink(config_path)

    def test_print_command_entry_matches_direct_train_model_call(self):
        with patch.object(
            textual_inversion_gui, "get_executable_path", return_value="accelerate"
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                kwargs = self._field_kwargs(
                    {"output_dir": tmpdir, "output_name": "wiring_test"}
                )
                data = self._field_data(kwargs)
                data[self.components["dummy_headless"]] = True
                data[self.components["dummy_db_true"]] = True
                mock_executor(textual_inversion_gui)
                self.entries["print_command"](data)
                toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
                self.assertEqual(len(toml_files), 1)
                with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                    via_wiring = toml.load(f)

            with tempfile.TemporaryDirectory() as tmpdir:
                kwargs = self._field_kwargs(
                    {"output_dir": tmpdir, "output_name": "wiring_test"}
                )
                mock_executor(textual_inversion_gui)
                textual_inversion_gui.train_model(
                    headless=True, print_only=True, **kwargs
                )
                toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
                with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                    via_direct_call = toml.load(f)

        for key in ("output_dir", "sample_prompts"):
            via_wiring.pop(key, None)
            via_direct_call.pop(key, None)
        self.assertEqual(via_wiring, via_direct_call)


if __name__ == "__main__":
    unittest.main()
