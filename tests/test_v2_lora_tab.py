"""Move 6 (checkpoint B7, LoRA portion): headless construction test for the
v2 LoRA tab. Gradio Blocks can be constructed without launching a server.
"""

import gradio as gr

from kohya_gui_v2.tabs.lora_fields import LORA_REGISTRY
from kohya_gui_v2.tabs.lora_tab import lora_tab


def test_lora_tab_constructs_with_one_component_per_non_architecture_field():
    with gr.Blocks():
        architecture, ordered_names, ordered_components = lora_tab(headless=True)

    expected_names = [n for n in LORA_REGISTRY.names() if n != "architecture"]
    assert ordered_names == expected_names
    assert len(ordered_components) == len(ordered_names)
