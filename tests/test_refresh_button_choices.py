"""Regression: create_refresh_button must keep gr.Dropdown.choices normalized.

The refresh helper copies the refreshed args onto the live component with
setattr(). For a Dropdown the choices arrive as a raw list of strings; assigning
them un-normalized leaves ``self.choices`` as plain strings, so the next
``gr.Dropdown.api_info()`` (which iterates choices as ``(label, value)`` pairs via
``[c[1] for c in self.choices]``) raises ``IndexError: string index out of range``
on an empty-string choice. This surfaced as a recurring console traceback after
clicking any refresh button (config file, model list, VAE, folders, ...).
"""

import os
import unittest

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from kohya_gui.common_gui import create_refresh_button


def _refresh_fn(demo, target):
    for f in demo.fns.values():
        fn = getattr(f, "fn", None)
        if fn is None:
            continue
        if getattr(fn, "__name__", "") == "refresh" and any(
            o is target for o in (f.outputs or [])
        ):
            return fn
    raise AssertionError("refresh handler not found")


class TestRefreshButtonChoices(unittest.TestCase):
    def _build(self, refreshed_args):
        with gr.Blocks() as demo:
            dd = gr.Dropdown(
                label="cfg",
                choices=["a"],
                value="a",
                allow_custom_value=True,
                interactive=True,
            )
            create_refresh_button(dd, lambda: None, refreshed_args, "rid")
        return demo, dd

    def test_string_choices_are_normalized_and_api_info_works(self):
        demo, dd = self._build(lambda: {"choices": ["", "x", "y"]})
        _refresh_fn(demo, dd)()
        self.assertTrue(
            all(isinstance(c, tuple) and len(c) == 2 for c in dd.choices),
            msg=f"choices not normalized: {dd.choices!r}",
        )
        info = dd.api_info()  # must not raise IndexError
        self.assertEqual(info["enum"], ["", "x", "y"])

    def test_empty_string_choice_does_not_crash_api_info(self):
        demo, dd = self._build(lambda: {"choices": [""]})
        _refresh_fn(demo, dd)()
        self.assertEqual(dd.api_info()["enum"], [""])

    def test_tuple_choices_preserved(self):
        demo, dd = self._build(lambda: {"choices": [("Label", "val"), ("L2", "v2")]})
        _refresh_fn(demo, dd)()
        self.assertEqual(dd.choices, [("Label", "val"), ("L2", "v2")])
        self.assertEqual(dd.api_info()["enum"], ["val", "v2"])

    def test_empty_choices_list(self):
        demo, dd = self._build(lambda: {"choices": []})
        _refresh_fn(demo, dd)()
        self.assertEqual(dd.choices, [])
        self.assertEqual(dd.api_info()["enum"], [])

    def test_returned_update_is_dropdown(self):
        demo, dd = self._build(lambda: {"choices": ["x"]})
        out = _refresh_fn(demo, dd)()
        self.assertIsInstance(out, gr.Dropdown)


if __name__ == "__main__":
    unittest.main()
