"""Regression test for GH issue #3520: dreambooth_gui.py leaking LoRA-only
`split_mode`/`train_blocks` into the flux_train.py (full fine-tune) config,
the same bug fixed in finetune_gui.py.

Also covers GH issue #3527: inpainting model training support (SD1.5/SDXL).
`--train_inpainting` must be forwarded to train_db.py/sdxl_train.py and must
never be combined with `--cache_latents`/`--cache_latents_to_disk` since
masks are generated randomly per step from the source image. It must also
never leak into flux_train.py/sd3_train.py, which aren't in the supported
script list.
"""

import unittest

from kohya_gui import dreambooth_gui
from conftest import (
    build_train_model_kwargs,
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


if __name__ == "__main__":
    unittest.main()
