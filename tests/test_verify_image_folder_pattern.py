"""GH #3374: DreamBooth/LoRA/TI image-folder preflight.

``verify_image_folder_pattern`` must reject common path mistakes before
sd-scripts starts and emits the opaque "No data found" error.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from kohya_gui import common_gui


def _touch_image(folder: str, name: str = "img.jpg") -> None:
    path = os.path.join(folder, name)
    with open(path, "wb") as fh:
        fh.write(b"\x00")


class TestVerifyImageFolderPattern(unittest.TestCase):
    def test_valid_layout_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            concept = os.path.join(tmp, "10_my_concept")
            os.makedirs(concept)
            _touch_image(concept)
            with patch.object(common_gui, "output_message") as mock_msg:
                ok = common_gui.verify_image_folder_pattern(tmp, headless=True)
            self.assertTrue(ok)
            mock_msg.assert_not_called()

    def test_spaced_concept_name_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            concept = os.path.join(tmp, "30_black mamba")
            os.makedirs(concept)
            _touch_image(concept)
            self.assertTrue(common_gui.verify_image_folder_pattern(tmp, headless=True))

    def test_leaf_folder_with_images_only_fails(self):
        """User pointed Image folder at the N_name leaf instead of its parent."""
        with tempfile.TemporaryDirectory() as tmp:
            leaf = os.path.join(tmp, "10_concept")
            os.makedirs(leaf)
            _touch_image(leaf)
            with patch.object(common_gui, "output_message") as mock_msg:
                ok = common_gui.verify_image_folder_pattern(leaf, headless=True)
            self.assertFalse(ok)
            msg = mock_msg.call_args.kwargs.get("msg") or mock_msg.call_args[0][0]
            self.assertIn("parent", msg.lower())
            self.assertIn("image_folder_structure", msg)

    def test_non_matching_subfolder_name_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "goth")
            os.makedirs(bad)
            _touch_image(bad)
            with patch.object(common_gui, "output_message") as mock_msg:
                ok = common_gui.verify_image_folder_pattern(tmp, headless=True)
            self.assertFalse(ok)
            msg = mock_msg.call_args.kwargs.get("msg") or mock_msg.call_args[0][0]
            self.assertTrue(
                any(s in msg for s in ("goth", "N_name", "repeats", "<number>")),
                msg=msg,
            )
            self.assertIn("image_folder_structure", msg)

    def test_empty_path_skipped(self):
        with patch.object(common_gui, "output_message") as mock_msg:
            ok = common_gui.verify_image_folder_pattern("", headless=True)
        self.assertTrue(ok)
        mock_msg.assert_not_called()

    def test_missing_path_fails(self):
        with patch.object(common_gui, "output_message") as mock_msg:
            ok = common_gui.verify_image_folder_pattern(
                os.path.join(tempfile.gettempdir(), "kohya_ss_no_such_dir_3374"),
                headless=True,
            )
        self.assertFalse(ok)
        mock_msg.assert_called()

    def test_matching_plus_extra_subfolder_still_passes(self):
        """At least one valid concept folder is enough; extras are warned only."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "10_concept"))
            os.makedirs(os.path.join(tmp, "notes"))
            self.assertTrue(common_gui.verify_image_folder_pattern(tmp, headless=True))


class TestTrainModelImageFolderPreflight(unittest.TestCase):
    """LoRA train_model aborts on bad layout and skips when dataset TOML is set."""

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
        "model_type",
        "model_prediction_type",
        "timestep_sampling",
        "train_blocks",
        "weighting_scheme",
        "in_dims",
        "base_weights",
        "base_weights_multiplier",
        "LyCORIS_preset",
        "network_args",
        "dataset_config",
        "log_tracker_config",
        "resume",
        "vae",
        "reg_data_dir",
    )

    def _kwargs(self, overrides):
        from conftest import build_train_model_kwargs
        from kohya_gui import lora_gui

        return build_train_model_kwargs(
            lora_gui.train_model,
            "test/config/locon-AdamW8bit-toml.json",
            numeric_fixups=self.NUMERIC_FIXUPS,
            string_overrides=self.STRING_OVERRIDES,
            overrides=overrides,
        )

    def test_lora_rejects_leaf_folder_before_launch(self):
        from conftest import mock_executor
        from kohya_gui import lora_gui

        with tempfile.TemporaryDirectory() as tmp:
            leaf = os.path.join(tmp, "10_concept")
            os.makedirs(leaf)
            _touch_image(leaf)
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            # Pretend a pretrained model path exists so earlier path checks pass.
            model = os.path.join(tmp, "model.safetensors")
            with open(model, "wb") as fh:
                fh.write(b"x")

            kwargs = self._kwargs(
                {
                    "print_only": True,
                    "train_data_dir": leaf,
                    "dataset_config": "",
                    "output_dir": out,
                    "logging_dir": os.path.join(tmp, "log"),
                    "pretrained_model_name_or_path": model,
                    "reg_data_dir": "",
                    "resume": "",
                    "vae": "",
                    "network_weights": "",
                    "log_tracker_config": "",
                    "LyCORIS_preset": "attn-mlp",
                }
            )
            mock_executor(lora_gui)
            with patch.object(
                lora_gui, "get_executable_path", return_value="accelerate"
            ):
                with patch.object(lora_gui, "validate_model_path", return_value=True):
                    with patch.object(common_gui, "output_message") as mock_msg:
                        with patch.object(lora_gui, "output_message", mock_msg):
                            result = lora_gui.train_model(**kwargs)

            self.assertIsNotNone(result)
            mock_msg.assert_called()
            msg = mock_msg.call_args.kwargs.get("msg") or mock_msg.call_args[0][0]
            self.assertIn("parent", msg.lower())

    def test_lora_skips_preflight_when_dataset_config_set(self):
        from conftest import mock_executor
        from kohya_gui import lora_gui

        with tempfile.TemporaryDirectory() as tmp:
            # Deliberately invalid DreamBooth layout (flat images).
            flat = os.path.join(tmp, "flat")
            os.makedirs(flat)
            _touch_image(flat)
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            model = os.path.join(tmp, "model.safetensors")
            with open(model, "wb") as fh:
                fh.write(b"x")
            toml_path = os.path.join(tmp, "dataset.toml")
            with open(toml_path, "w", encoding="utf-8") as fh:
                fh.write("[general]\n")

            kwargs = self._kwargs(
                {
                    "print_only": True,
                    "train_data_dir": flat,
                    "dataset_config": toml_path,
                    "output_dir": out,
                    "logging_dir": os.path.join(tmp, "log"),
                    "pretrained_model_name_or_path": model,
                    "reg_data_dir": "",
                    "resume": "",
                    "vae": "",
                    "network_weights": "",
                    "log_tracker_config": "",
                    "LyCORIS_preset": "attn-mlp",
                }
            )
            mock_executor(lora_gui)
            with patch.object(
                lora_gui, "get_executable_path", return_value="accelerate"
            ):
                with patch.object(lora_gui, "validate_model_path", return_value=True):
                    with patch.object(
                        lora_gui, "validate_file_path", return_value=True
                    ):
                        with patch.object(
                            lora_gui, "validate_toml_file", return_value=True
                        ):
                            with patch.object(
                                common_gui,
                                "verify_image_folder_pattern",
                            ) as mock_verify:
                                with patch.object(
                                    lora_gui, "write_toml_config", return_value=True
                                ):
                                    with patch.object(
                                        lora_gui,
                                        "print_command_and_toml",
                                    ):
                                        lora_gui.train_model(**kwargs)

            mock_verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
