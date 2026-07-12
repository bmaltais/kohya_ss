"""Finetune v2 tab — thin wrapper around shared ``build_training_tab``."""

from .finetune_derivations import derive
from .finetune_fields import ARCHITECTURE_CHOICES, FINETUNE_REGISTRY
from ..tab_builder import build_training_tab

ARCHITECTURE_SCRIPTS = {
    "base": "fine_tune.py",
    "sdxl": "sdxl_train.py",
    "sd3": "sd3_train.py",
    "flux1": "flux_train.py",
    "anima": "anima_train.py",
    "lumina": "lumina_train.py",
}


def finetune_tab(headless: bool = False, config=None):
    return build_training_tab(
        registry=FINETUNE_REGISTRY,
        derive=derive,
        training_type="finetune",
        arch_choices=ARCHITECTURE_CHOICES,
        arch_scripts=ARCHITECTURE_SCRIPTS,
        headless=headless,
        config=config,
        config_placeholder="./config_finetune_v2.toml",
        default_arch="base",
    )
