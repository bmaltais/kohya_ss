"""Anima LLLite v2 tab — thin wrapper around shared ``build_training_tab``.

Single-architecture training type; dropdown kept for registry-shape parity.
"""

from .anima_lllite_derivations import derive
from .anima_lllite_fields import ANIMA_LLLITE_REGISTRY, ARCHITECTURE_CHOICES
from ..tab_builder import build_training_tab

ARCHITECTURE_SCRIPTS = {
    "anima_lllite": "anima_train_control_net_lllite.py",
}


def anima_lllite_tab(headless: bool = False, config=None):
    return build_training_tab(
        registry=ANIMA_LLLITE_REGISTRY,
        derive=derive,
        training_type="anima_lllite",
        arch_choices=ARCHITECTURE_CHOICES,
        arch_scripts=ARCHITECTURE_SCRIPTS,
        headless=headless,
        config=config,
        config_placeholder="./config_anima_lllite_v2.toml",
        default_arch="anima_lllite",
    )
