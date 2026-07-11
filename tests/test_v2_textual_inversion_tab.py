"""Move 6 (checkpoint B7): headless construction test for the v2 Textual
Inversion tab. Gradio Blocks can be constructed without launching a server.
"""

import gradio as gr

from kohya_gui_v2.tabs.textual_inversion_fields import TEXTUAL_INVERSION_REGISTRY
from kohya_gui_v2.tabs.textual_inversion_tab import textual_inversion_tab


def test_textual_inversion_tab_constructs_with_one_component_per_non_architecture_field():
    with gr.Blocks():
        architecture, ordered_names, ordered_components = textual_inversion_tab(
            headless=True
        )

    expected_names = [
        n for n in TEXTUAL_INVERSION_REGISTRY.names() if n != "architecture"
    ]
    assert ordered_names == expected_names
    assert len(ordered_components) == len(ordered_names)
