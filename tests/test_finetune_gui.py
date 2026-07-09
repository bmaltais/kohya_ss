"""Regression test for GH issue #3520: finetune_gui.py leaking LoRA-only
`split_mode`/`train_blocks` into the flux_train.py (full fine-tune) config,
even though that script never defines either arg.

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

from kohya_gui import finetune_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from conftest import (
    build_train_model_kwargs,
    mock_executor,
    run_train_model_and_load_saved_json,
    run_train_model_and_load_toml,
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
)


class TestFinetuneFluxConfigOutput(unittest.TestCase):
    def test_split_mode_and_train_blocks_never_reach_flux_train_config(self):
        # Even if the (hidden, per class_flux1.py) fields somehow carry a
        # truthy value through to train_model, the fine-tune tab must never
        # forward them: it only ever targets flux_train.py, which doesn't
        # accept these LoRA-only (flux_train_network.py) args.
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={
                "flux1_checkbox": True,
                "split_mode": True,
                "train_blocks": "all",
            },
        )
        config = run_train_model_and_load_toml(finetune_gui, kwargs)

        self.assertNotIn("split_mode", config)
        self.assertNotIn("train_blocks", config)


class TestFinetuneTrainInpainting(unittest.TestCase):
    """GH issue #3527: inpainting model training support (SD1.5/SDXL).

    `--train_inpainting` must be forwarded to fine_tune.py/sdxl_train.py and
    must never be combined with `--cache_latents`/`--cache_latents_to_disk`
    since masks are generated randomly per step from the source image. It
    must also never leak into flux_train.py/sd3_train.py, which aren't in
    the supported script list.
    """

    def test_train_inpainting_forwarded_and_cache_latents_dropped(self):
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={
                "train_inpainting": True,
                "cache_latents": True,
                "cache_latents_to_disk": True,
            },
        )
        config = run_train_model_and_load_toml(finetune_gui, kwargs)

        self.assertTrue(config.get("train_inpainting"))
        self.assertNotIn("cache_latents", config)
        self.assertNotIn("cache_latents_to_disk", config)

    def test_train_inpainting_dropped_for_flux_backend(self):
        # train_inpainting is only supported by fine_tune.py/sdxl_train.py;
        # flux_train.py must never receive it.
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={
                "train_inpainting": True,
                "flux1_checkbox": True,
            },
        )
        config = run_train_model_and_load_toml(finetune_gui, kwargs)

        self.assertNotIn("train_inpainting", config)

    def test_saved_json_config_reflects_forced_overrides(self):
        # The JSON training config saved via SaveConfigFile() must not
        # persist the pre-override checkbox state (train_inpainting=True
        # with cache_latents=True), which would produce an invalid combo
        # if the user reloads this saved config later.
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={
                "train_inpainting": True,
                "cache_latents": True,
                "cache_latents_to_disk": True,
            },
        )
        config = run_train_model_and_load_saved_json(finetune_gui, kwargs)

        self.assertTrue(config.get("train_inpainting"))
        self.assertFalse(config.get("cache_latents"))
        self.assertFalse(config.get("cache_latents_to_disk"))


# train_model names differ from save/open for a few historical aliases.
_FINETUNE_TRAIN_TO_CONFIG = {
    "generate_caption_database": "create_caption",
    "generate_image_buckets": "create_buckets",
}


class TestFinetuneGuiFieldRegistry(unittest.TestCase):
    """GH #3543 M3: FIELD_REGISTRY must match train_model keyword order."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            finetune_gui.finetune_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = finetune_gui.last_built_field_registry

    def test_field_registry_was_built(self):
        self.assertIsNotNone(self.field_registry)
        self.assertGreater(len(self.field_registry), 0)

    def test_field_registry_names_are_unique(self):
        names = [name for name, _ in self.field_registry]
        self.assertEqual(len(names), len(set(names)))

    def test_field_registry_matches_train_model_signature(self):
        registry_names = [name for name, _ in self.field_registry]
        train_model_params = list(
            inspect.signature(finetune_gui.train_model).parameters
        )
        self.assertEqual(registry_names, train_model_params[2:])

    def test_field_registry_matches_save_configuration_after_aliases(self):
        registry_names = [
            _FINETUNE_TRAIN_TO_CONFIG.get(name, name) for name, _ in self.field_registry
        ]
        save_config_params = list(
            inspect.signature(finetune_gui.save_configuration).parameters
        )
        self.assertEqual(registry_names, save_config_params[2:])

    def test_field_registry_matches_open_configuration_after_aliases(self):
        registry_names = [
            _FINETUNE_TRAIN_TO_CONFIG.get(name, name) for name, _ in self.field_registry
        ]
        open_config_params = list(
            inspect.signature(finetune_gui.open_configuration).parameters
        )
        # open: ask_for_file, apply_preset, file_path, ...fields..., training_preset
        self.assertEqual(registry_names, open_config_params[3:-1])

    def test_sd3_registry_names_map_to_matching_widget_labels(self):
        # Regression for the finetune SD3 field association bug: the old settings_list
        # ordered clip_g before sd3_fused_backward_pass while train_model's
        # signature has fused first. Pairing by index then labeled fused as
        # "CLIP-G Path". Assert labels match the intended widgets.
        by_name = dict(self.field_registry)
        expected_label_substrings = {
            "sd3_fused_backward_pass": "Fused Backward Pass",
            "clip_g": "CLIP-G",
            "clip_l": "CLIP-L",
            "sd3_text_encoder_batch_size": "Text Encoder Batch Size",
            "weighting_scheme": "Weighting Scheme",
        }
        for name, needle in expected_label_substrings.items():
            label = getattr(by_name[name], "label", "") or ""
            self.assertIn(
                needle,
                label,
                msg=f"{name} mapped to unexpected widget label {label!r}",
            )


class TestFinetuneGuiDictAdapterWiring(unittest.TestCase):
    """GH #3543 M3: train/save/load buttons resolve args by component identity."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            finetune_gui.finetune_tab(headless=True, config=KohyaSSGUIConfig())
        cls.field_registry = finetune_gui.last_built_field_registry
        cls.entries = finetune_gui.last_built_gui_entries
        cls.components = cls.entries["components"]
        cls.settings_list = [comp for _, comp in cls.field_registry]

    def _field_data(self, kwargs: dict) -> dict:
        return {comp: kwargs[name] for name, comp in self.field_registry}

    def _field_kwargs(self, overrides=None):
        kwargs = build_train_model_kwargs(
            finetune_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=overrides,
        )
        kwargs.pop("headless")
        kwargs.pop("print_only")
        return kwargs

    def _config_kwargs(self, kwargs: dict) -> dict:
        return {_FINETUNE_TRAIN_TO_CONFIG.get(k, k): v for k, v in kwargs.items()}

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
            finetune_gui.save_configuration(
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
            finetune_gui.save_configuration(
                save_as_bool=False, file_path=config_path, **config_kwargs
            )
            data = self._field_data(kwargs)
            data[self.components["dummy_db_false"]] = False
            data[self.components["config_file_name"]] = config_path
            data[self.components["training_preset"]] = ""

            result = self.entries["load_configuration"](data)
            direct_result = finetune_gui.open_configuration(
                ask_for_file=False,
                apply_preset=False,
                file_path=config_path,
                training_preset="",
                **config_kwargs,
            )

            self.assertEqual(
                result[self.components["config_file_name"]], direct_result[0]
            )
            for comp, expected in zip(self.settings_list, direct_result[1:-1]):
                self.assertEqual(result[comp], expected)
            self.assertEqual(
                result[self.components["training_preset"]], direct_result[-1]
            )
        finally:
            os.unlink(config_path)

    def test_print_command_entry_matches_direct_train_model_call(self):
        with patch.object(
            finetune_gui, "get_executable_path", return_value="accelerate"
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                kwargs = self._field_kwargs(
                    {"output_dir": tmpdir, "output_name": "wiring_test"}
                )
                data = self._field_data(kwargs)
                data[self.components["dummy_headless"]] = True
                data[self.components["dummy_db_true"]] = True
                mock_executor(finetune_gui)
                self.entries["print_command"](data)
                toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
                self.assertEqual(len(toml_files), 1)
                with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                    via_wiring = toml.load(f)

            with tempfile.TemporaryDirectory() as tmpdir:
                kwargs = self._field_kwargs(
                    {"output_dir": tmpdir, "output_name": "wiring_test"}
                )
                mock_executor(finetune_gui)
                finetune_gui.train_model(headless=True, print_only=True, **kwargs)
                toml_files = [f for f in os.listdir(tmpdir) if f.endswith(".toml")]
                with open(os.path.join(tmpdir, toml_files[0]), encoding="utf-8") as f:
                    via_direct_call = toml.load(f)

        for key in ("output_dir", "sample_prompts"):
            via_wiring.pop(key, None)
            via_direct_call.pop(key, None)
        self.assertEqual(via_wiring, via_direct_call)


if __name__ == "__main__":
    unittest.main()
