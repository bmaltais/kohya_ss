import gradio as gr
import os
from typing import Tuple


class AccelerateLaunch:
    def __init__(
        self
    ) -> None:
        # self.sdxl_checkbox = sdxl_checkbox
        # self.learning_rate_value = learning_rate_value
        # self.lr_scheduler_value = lr_scheduler_value
        # self.lr_warmup_value = lr_warmup_value
        # self.finetuning = finetuning
        # self.dreambooth = dreambooth

        # Initialize the UI components
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        """
        Initializes the UI components for the training settings.
        """
        # Initialize the Hardware Selection Arguments
        self.init_hardware_selection_arguments()
        # Initialize the Resource Selection Arguments
        self.init_resource_selection_arguments()
        
    def init_hardware_selection_arguments(self) -> None:
        """
        Initializes the hardware selection arguments for the model.
        """
        with gr.Row():
            # Initialize the CPU checkbox
            self.cpu = gr.Checkbox(label="CPU", value=False, info="force the training on the CPU")
            # Initialize the multi-GPU checkbox
            self.multi_gpu = gr.Checkbox(label="Multi-GPU", value=False, info="launch a distributed GPU training")
            # Initialize the TPU checkbox
            self.tpu = gr.Checkbox(label="TPU", value=False, info="launch a TPU training")
            # Initialize the IPEX checkbox
            self.ipex = gr.Checkbox(label="IPEX", value=False, info="launch a Intel PyTorch Extension (IPEX) training")
