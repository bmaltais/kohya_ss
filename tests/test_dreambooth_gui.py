"""Regression test for GH issue #3520: dreambooth_gui.py leaking LoRA-only
`split_mode`/`train_blocks` into the flux_train.py (full fine-tune) config,
the same bug fixed in finetune_gui.py.
"""

import unittest

from kohya_gui import dreambooth_gui
from conftest import build_train_model_kwargs, run_train_model_and_load_toml

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


if __name__ == "__main__":
    unittest.main()
