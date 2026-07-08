"""Regression test for GH issue #3520: finetune_gui.py leaking LoRA-only
`split_mode`/`train_blocks` into the flux_train.py (full fine-tune) config,
even though that script never defines either arg.
"""

import unittest

from kohya_gui import finetune_gui
from conftest import build_train_model_kwargs, run_train_model_and_load_toml

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


if __name__ == "__main__":
    unittest.main()
