"""Move 6 (checkpoint B7): headless construction test for the v2 DreamBooth
tab. Gradio Blocks can be constructed without launching a server.
"""

import gradio as gr

from kohya_gui_v2.tabs.dreambooth_fields import DREAMBOOTH_REGISTRY
from kohya_gui_v2.tabs.dreambooth_tab import dreambooth_tab


def test_dreambooth_tab_constructs_with_one_component_per_non_architecture_field():
    with gr.Blocks():
        architecture, ordered_names, ordered_components = dreambooth_tab(headless=True)

    expected_names = [n for n in DREAMBOOTH_REGISTRY.names() if n != "architecture"]
    assert ordered_names == expected_names
    assert len(ordered_components) == len(ordered_names)
