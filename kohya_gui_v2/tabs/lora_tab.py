"""LoRA v2 tab — thin wrapper around shared ``build_training_tab``."""

from .lora_derivations import derive
from .lora_fields import ARCHITECTURE_CHOICES, LORA_REGISTRY
from ..tab_builder import build_training_tab

ARCHITECTURE_SCRIPTS = {
    "sd15": "train_network.py",
    "sd2": "train_network.py",
    "sdxl": "sdxl_train_network.py",
    "flux1": "flux_train_network.py",
    "sd3": "sd3_train_network.py",
    "hunyuan_image": "hunyuan_image_train_network.py",
    "anima": "anima_train_network.py",
    "lumina": "lumina_train_network.py",
}


def lora_tab(headless: bool = False, config=None):
    return build_training_tab(
        registry=LORA_REGISTRY,
        derive=derive,
        training_type="lora",
        arch_choices=ARCHITECTURE_CHOICES,
        arch_scripts=ARCHITECTURE_SCRIPTS,
        headless=headless,
        config=config,
        config_placeholder="./config_lora_v2.toml",
        default_arch="sd15",
    )
