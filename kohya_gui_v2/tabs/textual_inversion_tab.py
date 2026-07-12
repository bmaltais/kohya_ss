"""Textual Inversion v2 tab — thin wrapper around shared ``build_training_tab``."""

from .textual_inversion_derivations import derive
from .textual_inversion_fields import ARCHITECTURE_CHOICES, TEXTUAL_INVERSION_REGISTRY
from ..tab_builder import build_training_tab

ARCHITECTURE_SCRIPTS = {
    "sd_v1v2": "train_textual_inversion.py",
    "sdxl": "sdxl_train_textual_inversion.py",
}


def textual_inversion_tab(headless: bool = False, config=None):
    return build_training_tab(
        registry=TEXTUAL_INVERSION_REGISTRY,
        derive=derive,
        training_type="textual_inversion",
        arch_choices=ARCHITECTURE_CHOICES,
        arch_scripts=ARCHITECTURE_SCRIPTS,
        headless=headless,
        config=config,
        config_placeholder="./config_textual_inversion_v2.toml",
        default_arch="sd_v1v2",
    )
