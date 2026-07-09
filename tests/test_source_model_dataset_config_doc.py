"""Regression test for GH issue #3531: the GUI's dataset_config field should
point users to the upstream config_README-en.md section explaining that
multi-resolution / duplicate-subset datasets are configured via the TOML
file itself, not via GUI fields.
"""

import unittest

import gradio as gr

from kohya_gui.class_source_model import SourceModel


class TestDatasetConfigDocLink(unittest.TestCase):
    def test_dataset_config_info_links_to_duplicate_subsets_doc(self):
        with gr.Blocks():
            source_model = SourceModel(headless=True)

        info = source_model.dataset_config.info
        self.assertIsNotNone(info)
        self.assertIn(
            "config_README-en.md#behavior-when-there-are-duplicate-subsets", info
        )


if __name__ == "__main__":
    unittest.main()
