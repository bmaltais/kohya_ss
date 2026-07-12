"""DreamBooth v2 tab — thin wrapper around shared ``build_training_tab``."""

from .dreambooth_derivations import derive
from .dreambooth_fields import ARCHITECTURE_CHOICES, DREAMBOOTH_REGISTRY
from ..tab_builder import build_training_tab

ARCHITECTURE_SCRIPTS = {
    "sd15": "train_db.py",
    "sd2": "train_db.py",
    "sdxl": "sdxl_train.py",
    "sd3": "sd3_train.py",
    "flux1": "flux_train.py",
}


def dreambooth_tab(headless: bool = False, config=None):
    return build_training_tab(
        registry=DREAMBOOTH_REGISTRY,
        derive=derive,
        training_type="dreambooth",
        arch_choices=ARCHITECTURE_CHOICES,
        arch_scripts=ARCHITECTURE_SCRIPTS,
        headless=headless,
        config=config,
        config_placeholder="./config_dreambooth_v2.toml",
        default_arch="sd15",
    )
