"""Regression tests for MOSS-Image (OpenMOSS) LoRA training support in the GUI.

MOSS-Image is trained via `sd-scripts/moss_image_train_network.py` (sd-scripts
>= 0.12.1) with the generic `networks.lora_generic` module, which auto-detects
the MOSS-Image DiT architecture. Follows the Lumina/Anima/HunyuanImage model
family pattern: source-model checkbox + class_moss_image widget + command/TOML
wiring.
"""

import unittest
from unittest.mock import patch

from kohya_gui import lora_gui
from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
    mock_executor,
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
    "moss_image_max_token_length",
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
    "moss_image_text_encoder",
    "moss_image_vae",
    "moss_image_tokenizer_cache_dir",
    "moss_image_timestep_sampling",
    "moss_image_attn_mode",
    "moss_image_convrot_int8_bwd",
    "moss_image_sample_uncond_skip_layers",
)

MOSS_OVERRIDES = {
    "LoRA_type": "MOSS-Image",
    "moss_image_checkbox": True,
    "moss_image_text_encoder": "/models/qwen3.5_9b.safetensors",
    "moss_image_vae": "/models/flux2-vae.safetensors",
    "moss_image_max_token_length": 4096,
    "moss_image_discrete_flow_shift": 1.0,
    "moss_image_timestep_sampling": "sigmoid",
    "moss_image_sigmoid_scale": 1.0,
    "moss_image_attn_mode": "torch",
    "moss_image_split_attn": False,
    "moss_image_convrot_int8": False,
    "moss_image_convrot_int8_text_encoder": False,
    "moss_image_convrot_int8_bwd": "bf16",
    "moss_image_cache_text_encoder_outputs": True,
    "moss_image_cache_text_encoder_outputs_to_disk": False,
}


class TestMossImageLoraConfigOutput(unittest.TestCase):
    """train_model(print_only=True) writes a real TOML we inspect for the
    MOSS-Image specific keys."""

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**MOSS_OVERRIDES, **overrides},
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_network_module_is_lora_generic(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("network_module"), "networks.lora_generic")

    def test_model_paths_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("text_encoder"), "/models/qwen3.5_9b.safetensors")
        self.assertEqual(config.get("vae"), "/models/flux2-vae.safetensors")

    def test_max_token_length_uses_moss_key(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("moss_max_token_length"), 4096)
        # the SD-style max_token_length must not leak through
        self.assertNotIn("max_token_length", config)

    def test_flow_matching_params_are_forwarded(self):
        config = self._run_and_load_toml(
            {
                "moss_image_discrete_flow_shift": 3.0,
                "moss_image_timestep_sampling": "truncated_logit_normal",
                "moss_image_sigmoid_scale": 2.0,
            }
        )
        self.assertEqual(config.get("discrete_flow_shift"), 3.0)
        self.assertEqual(config.get("timestep_sampling"), "truncated_logit_normal")
        self.assertEqual(config.get("sigmoid_scale"), 2.0)

    def test_text_encoder_cache_flags_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertTrue(config.get("cache_text_encoder_outputs"))
        self.assertNotIn("cache_text_encoder_outputs_to_disk", config)

    def test_convrot_int8_flags_default_off(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("convrot_int8", config)
        self.assertNotIn("convrot_int8_text_encoder", config)
        self.assertNotIn("convrot_int8_bwd", config)

    def test_convrot_int8_flags_forwarded_when_set(self):
        config = self._run_and_load_toml(
            {
                "moss_image_convrot_int8": True,
                "moss_image_convrot_int8_text_encoder": True,
                "moss_image_convrot_int8_bwd": "int8",
            }
        )
        self.assertTrue(config.get("convrot_int8"))
        self.assertTrue(config.get("convrot_int8_text_encoder"))
        self.assertEqual(config.get("convrot_int8_bwd"), "int8")

    def test_convrot_int8_bwd_omitted_without_convrot_base(self):
        # bwd=int8 only matters with a ConvRot base; must not be emitted alone
        config = self._run_and_load_toml({"moss_image_convrot_int8_bwd": "int8"})
        self.assertNotIn("convrot_int8_bwd", config)

    def test_attn_mode_torch_is_omitted(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("attn_mode", config)

    def test_attn_mode_forwarded_when_not_torch(self):
        config = self._run_and_load_toml(
            {"moss_image_attn_mode": "flash", "moss_image_split_attn": True}
        )
        self.assertEqual(config.get("attn_mode"), "flash")
        self.assertTrue(config.get("split_attn"))

    def test_tokenizer_cache_dir_forwarded_when_set(self):
        config = self._run_and_load_toml(
            {"moss_image_tokenizer_cache_dir": "/models/moss_tokenizer"}
        )
        self.assertEqual(config.get("tokenizer_cache_dir"), "/models/moss_tokenizer")

    def test_tokenizer_cache_dir_omitted_when_empty(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("tokenizer_cache_dir", config)

    def test_incompatible_sd_args_are_excluded(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("max_token_length", config)
        self.assertNotIn("clip_skip", config)

    def test_fp8_base_excluded_for_moss(self):
        # MOSS-Image does not support fp8; ConvRot INT8 is the alternative.
        config = self._run_and_load_toml({"fp8_base": True})
        self.assertNotIn("fp8_base", config)

    def test_ddpm_only_loss_args_suppressed_even_with_stale_values(self):
        # A config migrated from SD/SDXL may carry values for args the MOSS
        # trainer ignores (post_process_loss is a no-op; the flow noise path
        # does not apply noise_offset/multires/adaptive). The emitted TOML
        # must not contain them.
        config = self._run_and_load_toml(
            {
                "debiased_estimation_loss": True,
                "min_snr_gamma": 5,
                "v_pred_like_loss": 0.5,
                "scale_v_pred_loss_like_noise_pred": True,
                "noise_offset": 0.1,
                "noise_offset_random_strength": True,
                "adaptive_noise_scale": 0.1,
                "multires_noise_iterations": 6,
                "multires_noise_discount": 0.3,
            }
        )
        for key in (
            "debiased_estimation_loss",
            "min_snr_gamma",
            "v_pred_like_loss",
            "scale_v_pred_loss_like_noise_pred",
            "noise_offset",
            "noise_offset_random_strength",
            "adaptive_noise_scale",
            "multires_noise_iterations",
            "multires_noise_discount",
        ):
            self.assertNotIn(key, config)

    def test_ip_noise_gamma_still_forwarded_for_moss(self):
        # ip_noise_gamma IS applied by the flow-matching noise path
        # (flux_train_utils.get_noisy_model_input_and_timesteps), so unlike
        # the other noise args it must survive.
        config = self._run_and_load_toml({"ip_noise_gamma": 0.1})
        self.assertEqual(config.get("ip_noise_gamma"), 0.1)

    def test_sample_uncond_skip_layers_default_omitted(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("sample_uncond_skip_layers", config)

    def test_sample_uncond_skip_layers_forwarded_when_changed(self):
        config = self._run_and_load_toml({"moss_image_sample_uncond_skip_layers": "10"})
        self.assertEqual(config.get("sample_uncond_skip_layers"), "10")

    def test_script_selection_invokes_moss_image_train_network(self):
        with patch.object(lora_gui, "print_command_and_toml") as mocked:
            kwargs = build_train_model_kwargs(
                lora_gui.train_model,
                FIXTURE,
                numeric_fixups=NUMERIC_FIXUPS,
                string_overrides=STRING_OVERRIDES,
                overrides=MOSS_OVERRIDES,
            )
            mock_executor(lora_gui)
            with patch.object(
                lora_gui, "get_executable_path", return_value="accelerate"
            ):
                lora_gui.train_model(**kwargs)
            self.assertTrue(mocked.called)
            run_cmd = mocked.call_args[0][0]
            self.assertTrue(
                any("moss_image_train_network.py" in part for part in run_cmd)
            )


class TestMossImageValidation(unittest.TestCase):
    """train_model must reject a MOSS-Image LoRA type without the checkbox and
    vice-versa, mirroring the Lumina/Anima guards."""

    def _run(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**MOSS_OVERRIDES, **overrides},
        )
        mock_executor(lora_gui)
        with patch.object(lora_gui, "get_executable_path", return_value="accelerate"):
            with patch.object(lora_gui, "print_command_and_toml") as mocked:
                lora_gui.train_model(**kwargs)
        return mocked.called

    def test_lora_type_without_checkbox_is_rejected(self):
        self.assertFalse(self._run({"moss_image_checkbox": False}))

    def test_checkbox_without_lora_type_is_rejected(self):
        self.assertFalse(self._run({"LoRA_type": "Standard"}))

    def test_matching_type_and_checkbox_is_accepted(self):
        self.assertTrue(self._run({}))


class TestMossImageNativeLohaLokr(unittest.TestCase):
    """Kohya LoHa/LoKr auto-detect MOSS-Image; the checkbox pair must be
    accepted and route to moss_image_train_network.py."""

    def _run(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**MOSS_OVERRIDES, **overrides},
        )
        mock_executor(lora_gui)
        with patch.object(lora_gui, "get_executable_path", return_value="accelerate"):
            with patch.object(lora_gui, "print_command_and_toml") as mocked:
                lora_gui.train_model(**kwargs)
        return mocked

    def test_kohya_lokr_with_moss_checkbox_routes_to_moss_script(self):
        mocked = self._run({"LoRA_type": "Kohya LoKr"})
        self.assertTrue(mocked.called)
        run_cmd = mocked.call_args[0][0]
        self.assertTrue(any("moss_image_train_network.py" in p for p in run_cmd))


class TestMossImageAdvancedFieldVisibility(unittest.TestCase):
    """The LoRA-type change handler must hide the Advanced controls that the
    MOSS-Image trainer ignores, and show them again for every other type.

    Hidden for MOSS-Image: the whole noise-offset row (noise_offset_type,
    noise_offset, multires, adaptive_noise_scale, ip_noise_gamma), clip_skip,
    max_token_length, min_snr_gamma, debiased_estimation_loss,
    v_pred_like_loss, scale_v_pred_loss_like_noise_pred.
    """

    @classmethod
    def setUpClass(cls):
        import gradio as gr

        from kohya_gui.class_gui_config import KohyaSSGUIConfig

        with gr.Blocks() as demo:
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        cls.dep_fn = None
        for fn in demo.fns.values():
            if getattr(fn.fn, "__name__", "") == "update_LoRA_settings":
                cls.dep_fn = fn
                break
        assert cls.dep_fn is not None, "update_LoRA_settings not wired"

    def _visibility(self, lora_type, moss_image_checkbox=False):
        """Return {label-or-component: visible} for handler outputs."""
        results = self.dep_fn.fn(lora_type, moss_image_checkbox)
        assert len(results) == len(self.dep_fn.outputs)
        vis = {}
        for update, comp in zip(results, self.dep_fn.outputs):
            label = getattr(comp, "label", None) or type(comp).__name__
            vis[label] = update.constructor_args.get("visible")
        return vis

    HIDDEN_FOR_MOSS = (
        "Clip skip",
        "Max Token Length",
        "Min SNR gamma",
        "Debiased Estimation loss",
        "V Pred like loss",
        "Scale v prediction loss",
    )

    def test_moss_hides_ddpm_only_advanced_controls(self):
        vis = self._visibility("MOSS-Image", True)
        for label in self.HIDDEN_FOR_MOSS:
            self.assertIs(
                vis.get(label), False, f"{label} must be hidden for MOSS-Image"
            )

    def test_moss_hides_noise_offset_row(self):
        # The wiring has [..., loraplus, noise_offset_row, clip_skip, ...]:
        # the noise-offset Row is the bare gr.Row right before the Clip skip
        # output (it has no label to key on, and its position from the end
        # changes whenever another output is appended).
        clip_skip_index = [
            i
            for i, c in enumerate(self.dep_fn.outputs)
            if getattr(c, "label", None) == "Clip skip"
        ][0]
        row_index = clip_skip_index - 1
        self.assertEqual(type(self.dep_fn.outputs[row_index]).__name__, "Row")
        results = self.dep_fn.fn("MOSS-Image", True)
        self.assertIs(
            results[row_index].constructor_args.get("visible"),
            False,
            "noise_offset_row must be hidden for MOSS-Image",
        )
        results_std = self.dep_fn.fn("Standard", False)
        self.assertIs(
            results_std[row_index].constructor_args.get("visible"),
            True,
            "noise_offset_row must be visible again for Standard",
        )

    # The MOSS-Image accordion owns the VAE / attention mode settings, so the
    # Advanced duplicates must be hidden for every MOSS-compatible LoRA type.
    HIDDEN_FOR_MOSS_DUPLICATES = (
        "VAE (Optional: Path to checkpoint of vae for training)",
        "Memory efficient attention",
        "CrossAttention",
    )

    def test_moss_hides_advanced_vae_and_attention_duplicates(self):
        for lora_type, moss in (("MOSS-Image", True), ("Kohya LoHa", True)):
            vis = self._visibility(lora_type, moss)
            for label in self.HIDDEN_FOR_MOSS_DUPLICATES:
                self.assertIs(
                    vis.get(label),
                    False,
                    f"{label} must be hidden while MOSS-Image is selected",
                )
        vis = self._visibility("Standard", False)
        for label in self.HIDDEN_FOR_MOSS_DUPLICATES:
            self.assertIs(
                vis.get(label),
                True,
                f"{label} must stay visible without MOSS-Image",
            )

    def test_moss_hides_vae_buttons_with_the_dropdown(self):
        clip_skip_index = [
            i
            for i, c in enumerate(self.dep_fn.outputs)
            if getattr(c, "label", None) == "Clip skip"
        ][0]
        # clip_skip is followed by min_snr_gamma, debiased, v_pred,
        # scale_v_pred, then the VAE dropdown / refresh / open buttons.
        vae_outputs = slice(clip_skip_index + 5, clip_skip_index + 8)
        results = self.dep_fn.fn("MOSS-Image", True)
        for update, comp in zip(results[vae_outputs], self.dep_fn.outputs[vae_outputs]):
            self.assertIs(
                update.constructor_args.get("visible"),
                False,
                f"{type(comp).__name__} of the Advanced VAE control must be hidden",
            )

    def test_other_types_keep_advanced_controls_visible(self):
        for lora_type in ("Standard", "Kohya LoCon", "LyCORIS/LoCon", "Flux1"):
            vis = self._visibility(lora_type, False)
            for label in self.HIDDEN_FOR_MOSS:
                self.assertIs(
                    vis.get(label),
                    True,
                    f"{label} must stay visible for {lora_type}",
                )

    def test_handler_output_arity_matches_wiring(self):
        results = self.dep_fn.fn("MOSS-Image", True)
        self.assertEqual(len(results), len(self.dep_fn.outputs))


class TestMossImageLoRATypeChoices(unittest.TestCase):
    """With the MOSS-Image source model selected, the LoRA type dropdown must
    only offer what sd-scripts can train on a MossImageBlock DiT
    (networks.lora_generic, networks.loha, networks.lokr).

    networks.lora ("Standard") and lycoris.kohya match modules by their
    SD/Flux names and create 0 modules on MOSS-Image, so they are excluded.
    """

    @classmethod
    def setUpClass(cls):
        import gradio as gr

        from kohya_gui.class_gui_config import KohyaSSGUIConfig

        with gr.Blocks() as demo:
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        cls.dep_fn = None
        for fn in demo.fns.values():
            if getattr(fn.fn, "__name__", "") == "update_lora_type_choices":
                cls.dep_fn = fn
                break
        assert cls.dep_fn is not None, "update_lora_type_choices not wired"

    def test_moss_checkbox_restricts_choices(self):
        update = self.dep_fn.fn(True)
        self.assertEqual(
            list(update.constructor_args["choices"]),
            lora_gui.MOSS_IMAGE_LORA_TYPE_CHOICES,
        )
        self.assertEqual(update.constructor_args["value"], "MOSS-Image")

    def test_handler_reads_only_the_checkbox(self):
        # GH #3606: reading LoRA_type here would act on the pre-load snapshot
        # when a config load sets the checkbox and LoRA_type in one response,
        # and the echoed value would clobber the just-loaded type.
        self.assertEqual(len(self.dep_fn.inputs), 1)
        self.assertEqual(self.dep_fn.inputs[0].label, "MOSS-Image")

    def test_unchecking_restores_all_choices_without_echoing_a_value(self):
        update = self.dep_fn.fn(False)
        self.assertEqual(
            list(update.constructor_args["choices"]), lora_gui.LORA_TYPE_CHOICES
        )
        self.assertNotIn("value", update.constructor_args)


if __name__ == "__main__":
    unittest.main()
