"""Wargame Move 1 recon: does sd-scripts' --config_file loader tolerate
extra, unknown TOML keys?

The planned kohya_gui_v2 registry saves GUI-only bookkeeping keys
(`architecture`, `training_type`) directly into the same flat TOML that is
passed to `--config_file`. `read_config_from_file` (sd-scripts/library/args.py)
flattens all TOML sections into one dict and calls
`parser.parse_args(namespace=argparse.Namespace(**flat_dict))`. argparse does
not validate namespace attributes that were not added via `add_argument`, so
unknown keys should simply become extra, unused attributes on the resulting
Namespace instead of raising.

This test proves that assumption for every trainer family kohya_gui_v2 must
support in Phase A. If any parser here fails, Fork R1 in the wargame plan
(wargame/2026-07-11-musubi-style-gui-v2.md) activates: the saved config keeps
GUI-only keys, and train-time writes a filtered sibling copy instead.
"""

import argparse
import importlib
import os
import sys
import tempfile
from unittest.mock import patch

import pytest
import toml

pytest.importorskip(
    "torch", reason="sd-scripts entry modules import torch at module load time"
)

SD_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sd-scripts")


# Some parsers have CLI-required args unrelated to the config-file tolerance
# question (e.g. LeCo's --prompts_file: required=True is checked against argv
# presence, not the merged TOML namespace, so it must be passed directly on
# the command line even when --config_file is also used). Supply the minimal
# extra argv needed per module so the tolerance question can be isolated.
REQUIRED_CLI_EXTRAS = {
    "sdxl_train_leco": ["--prompts_file"],  # value filled in with a temp path
}


def _read_config_from_file(module_name: str, extra_junk: dict):
    """Import `module_name` from sd-scripts, build its parser, feed a minimal
    TOML containing `extra_junk` keys through read_config_from_file, and
    return the resulting argparse.Namespace.
    """
    if SD_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SD_SCRIPTS_DIR)

    module = importlib.import_module(module_name)
    from library.args import read_config_from_file

    parser = module.setup_parser()

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.toml")
        with open(config_path, "w", encoding="utf-8") as f:
            toml.dump(extra_junk, f)

        required_argv = []
        for flag in REQUIRED_CLI_EXTRAS.get(module_name, []):
            required_argv.extend([flag, os.path.join(tmpdir, "dummy.toml")])

        # read_config_from_file's second parser.parse_args(namespace=...) call
        # (sd-scripts/library/args.py:1183) does not take an explicit argv
        # list, so it re-reads sys.argv. Under pytest that would be pytest's
        # own CLI args, not ours -- pin argv to just what the trainer would
        # actually see on the command line for this run.
        argv = ["--config_file", config_path] + required_argv
        fake_argv = ["prog"] + argv
        with patch.object(sys, "argv", fake_argv):
            args = parser.parse_args(argv)
            return read_config_from_file(args, parser)


# (module name, any extra parser-specific defaults needed to avoid unrelated
# argparse errors unrelated to the tolerance question)
PARSER_MODULES = [
    "train_network",
    "train_db",
    "sdxl_train",
    "train_textual_inversion",
    "sdxl_train_leco",
    "anima_train_control_net_lllite",
]


@pytest.mark.parametrize("module_name", PARSER_MODULES)
def test_extra_toml_keys_are_tolerated(module_name):
    """A TOML config carrying GUI-only bookkeeping keys must not crash
    read_config_from_file's parse for any of the six trainer families."""
    extra_junk = {
        "architecture": "sdxl",
        "training_type": "lora",
        "totally_unknown_key_xyz": 123,
    }

    args = _read_config_from_file(module_name, extra_junk)

    # Unknown keys become plain Namespace attributes; argparse does not
    # reject them. Confirm they survived rather than being silently dropped
    # (dropping would also be "tolerant" but would contradict our round-trip
    # assumption that saved keys reappear on load).
    assert getattr(args, "architecture", None) == "sdxl"
    assert getattr(args, "training_type", None) == "lora"
    assert getattr(args, "totally_unknown_key_xyz", None) == 123
