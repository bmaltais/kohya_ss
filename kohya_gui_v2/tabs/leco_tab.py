"""LeCo v2 tab — thin wrapper around shared ``build_training_tab``.

Command construction appends REQUIRED_CLI_FIELDS (e.g. --prompts_file) on
argv because train_leco.py declares them required=True and argparse enforces
them before --config_file is read.
"""

from .leco_derivations import derive
from .leco_fields import ARCHITECTURE_CHOICES, LECO_REGISTRY, REQUIRED_CLI_FIELDS
from ..tab_builder import build_training_tab

ARCHITECTURE_SCRIPTS = {
    "sd15": "train_leco.py",
    "sd2": "train_leco.py",
    "sdxl": "sdxl_train_leco.py",
}


def leco_tab(headless: bool = False, config=None):
    return build_training_tab(
        registry=LECO_REGISTRY,
        derive=derive,
        training_type="leco",
        arch_choices=ARCHITECTURE_CHOICES,
        arch_scripts=ARCHITECTURE_SCRIPTS,
        headless=headless,
        config=config,
        config_placeholder="./config_leco_v2.toml",
        default_arch="sd15",
        required_cli_fields=REQUIRED_CLI_FIELDS,
    )
