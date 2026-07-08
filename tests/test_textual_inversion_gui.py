"""Regression test for GH issue #3520: textual_inversion_gui.py emitting
`stop_text_encoder_training`, an arg only defined by train_db.py — neither
train_textual_inversion.py nor sdxl_train_textual_inversion.py accept it.
"""

import unittest

from kohya_gui import textual_inversion_gui
from conftest import build_train_model_kwargs, run_train_model_and_load_toml

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


if __name__ == "__main__":
    unittest.main()
