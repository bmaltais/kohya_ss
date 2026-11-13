"""Regression tests for GH issues #3606 / #1846: loading a LyCORIS config
file must not reset Network Rank (Dimension) / Convolution Rank (Dimension).

Root cause (fixed): the ``LoRA_type.change`` handler (``update_LoRA_settings``)
used to read both rank sliders as *inputs* and echo their snapshot back via
``gr.Slider(maximum=..., value=<snapshot>)``. ``open_configuration`` sets
``LoRA_type`` and the rank sliders in ONE event response; the queued change
handler then read the PRE-load slider values and overwrote the freshly loaded
ones. The config only "worked" when loaded a second time (dropdown unchanged
→ no handler run), matching the two-load workaround reported in #1846.

The fix removes the slider inputs entirely and returns maximum-only updates,
so Gradio keeps whatever value the load event set (clamping client-side when
the maximum shrinks). These tests assert:

1. the handler's inputs are [LoRA_type, MOSS-Image checkbox] (no value
   snapshot race),
2. conv_dim / network_dim updates carry `maximum` but never `value`,
3. the extended maximum still applies to the rank-unbounded LyCORIS types,
4. open_configuration round-trips non-default ranks for LyCORIS configs.
"""

import json
import os
import tempfile
import unittest

import gradio as gr

from kohya_gui import lora_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig

RANK_UNBOUNDED_TYPES = {
    "LyCORIS/LoHa",
    "LyCORIS/LoKr",
    "LyCORIS/BOFT",
    "LyCORIS/Diag-OFT",
}
RANK_LABELS = ("Network Rank (Dimension)", "Convolution Rank (Dimension)")


def _build_tab():
    """Build the real LoRA tab and return (demo, the update_LoRA_settings fn)."""
    with gr.Blocks() as demo:
        lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
    for fn in demo.fns.values():
        if getattr(fn.fn, "__name__", "") == "update_LoRA_settings":
            return demo, fn
    raise AssertionError("update_LoRA_settings dependency not found")


class TestUpdateLoraSettingsNoValueEcho(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo, cls.dep_fn = _build_tab()

    def test_handler_inputs_do_not_snapshot_ranks(self):
        # conv_dim/network_dim must NOT be inputs: reading them snapshots the
        # pre-load value and reintroduces the #3606 race. The LoRA type and
        # the MOSS-Image source-model checkbox (a boolean, no rank snapshot)
        # are the only inputs.
        self.assertEqual(len(self.dep_fn.inputs), 2)
        self.assertEqual(self.dep_fn.inputs[0].label, "LoRA type")
        self.assertEqual(self.dep_fn.inputs[1].label, "MOSS-Image")
        input_labels = {getattr(c, "label", None) for c in self.dep_fn.inputs}
        self.assertFalse(input_labels & set(RANK_LABELS))

    def _rank_updates(self, lora_type):
        results = self.dep_fn.fn(lora_type, False)
        self.assertEqual(len(results), len(self.dep_fn.outputs))
        updates = {}
        for update, comp in zip(results, self.dep_fn.outputs):
            label = getattr(comp, "label", None)
            if label in RANK_LABELS:
                updates[label] = update
        self.assertEqual(set(updates), set(RANK_LABELS))
        return updates

    def test_rank_updates_never_echo_value(self):
        for lora_type in (
            "Standard",
            "LyCORIS/LoCon",
            "LyCORIS/LoHa",
            "LyCORIS/LoKr",
            "LyCORIS/BOFT",
            "LyCORIS/Diag-OFT",
            "Kohya LoCon",
            "Anima",
            "Flux1",
            "Lumina",
        ):
            for label, update in self._rank_updates(lora_type).items():
                self.assertNotIn(
                    "value",
                    update.constructor_args,
                    f"{lora_type}: {label} update must not set `value` (#3606)",
                )

    def test_maximum_is_extended_for_rank_unbounded_types(self):
        for lora_type in RANK_UNBOUNDED_TYPES:
            for label, update in self._rank_updates(lora_type).items():
                self.assertEqual(
                    update.constructor_args.get("maximum"),
                    100000,
                    f"{lora_type}: {label} maximum must stay 100000",
                )

    def test_maximum_is_512_for_other_types(self):
        for lora_type in ("Standard", "LyCORIS/LoCon", "Kohya LoCon", "LoRA-FA"):
            for label, update in self._rank_updates(lora_type).items():
                self.assertEqual(
                    update.constructor_args.get("maximum"),
                    512,
                    f"{lora_type}: {label} maximum must stay 512",
                )

    def test_output_arity_matches_wiring(self):
        # The handler still returns one update per wired output (28 today);
        # Gradio raises if an event returns fewer values than outputs.
        results = self.dep_fn.fn("LyCORIS/LoCon", False)
        self.assertEqual(len(results), len(self.dep_fn.outputs))


class TestOpenConfigurationKeepsRanks(unittest.TestCase):
    """Server side of the same bug: a saved LyCORIS config with non-default
    ranks must round-trip through open_configuration (the values the load
    event pushes into the sliders in the same response as LoRA_type)."""

    def _build_kwargs(self):
        import inspect

        sig = inspect.signature(lora_gui.open_configuration)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name in ("ask_for_file", "apply_preset", "file_path"):
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            kwargs[name] = None
        kwargs["training_preset"] = "none"
        return kwargs

    def test_lycoris_ranks_survive_load(self):
        config = {
            "LoRA_type": "LyCORIS/LoCon",
            "network_dim": 64,
            "conv_dim": 32,
            "network_alpha": 16,
            "conv_alpha": 8,
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            result = lora_gui.open_configuration(
                ask_for_file=False,
                apply_preset=False,
                file_path=path,
                **self._build_kwargs(),
            )
        finally:
            os.unlink(path)

        self.assertIsNotNone(result)
        # open_configuration returns [file_path, *field-order values,
        # training_preset, convolution_row]; locate by FIELD_REGISTRY order.
        import inspect

        param_names = list(inspect.signature(lora_gui.open_configuration).parameters)
        param_names = [
            p
            for p in param_names
            if p not in ("ask_for_file", "apply_preset", "file_path")
        ]
        # result = [file_path, *param-order values, convolution_row]
        by_name = dict(zip(param_names, result[1:-1]))
        self.assertEqual(by_name["LoRA_type"], "LyCORIS/LoCon")
        self.assertEqual(by_name["network_dim"], 64)
        self.assertEqual(by_name["conv_dim"], 32)
        self.assertEqual(by_name["network_alpha"], 16)
        self.assertEqual(by_name["conv_alpha"], 8)
        # convolution_row must be visible for LyCORIS/LoCon
        self.assertTrue(result[-1].visible)

    def test_loha_big_ranks_survive_load(self):
        config = {
            "LoRA_type": "LyCORIS/LoHa",
            "network_dim": 1024,
            "conv_dim": 1024,
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            result = lora_gui.open_configuration(
                ask_for_file=False,
                apply_preset=False,
                file_path=path,
                **self._build_kwargs(),
            )
        finally:
            os.unlink(path)

        self.assertIsNotNone(result)
        import inspect

        param_names = [
            p
            for p in inspect.signature(lora_gui.open_configuration).parameters
            if p not in ("ask_for_file", "apply_preset", "file_path")
        ]
        by_name = dict(zip(param_names, result[1:-1]))
        self.assertEqual(by_name["network_dim"], 1024)
        self.assertEqual(by_name["conv_dim"], 1024)


if __name__ == "__main__":
    unittest.main()
