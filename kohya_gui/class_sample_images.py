import os
import gradio as gr
import shlex

from .custom_logging import setup_logging
from .class_gui_config import KohyaSSGUIConfig

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄


###
### Gradio common sampler GUI section
###
def create_prompt_file(sample_prompts, output_dir):
    """
    Creates a prompt file for image sampling.

    Args:
        sample_prompts (str): The prompts to use for image sampling.
        output_dir (str): The directory where the output images will be saved.

    Returns:
        str: The path to the prompt file.
    """
    sample_prompts_path = os.path.join(output_dir, "sample/prompt.txt")

    if not os.path.exists(os.path.dirname(sample_prompts_path)):
        os.makedirs(os.path.dirname(sample_prompts_path))

    with open(sample_prompts_path, "w", encoding="utf-8") as f:
        f.write(sample_prompts)

    return sample_prompts_path


# def run_cmd_sample(
#     run_cmd: list,
#     sample_every_n_steps,
#     sample_every_n_epochs,
#     sample_sampler,
#     sample_prompts,
#     output_dir,
# ):
#     """
#     Generates a command string for sampling images during training.

#     Args:
#         sample_every_n_steps (int): The number of steps after which to sample images.
#         sample_every_n_epochs (int): The number of epochs after which to sample images.
#         sample_sampler (str): The sampler to use for image sampling.
#         sample_prompts (str): The prompts to use for image sampling.
#         output_dir (str): The directory where the output images will be saved.

#     Returns:
#         str: The command string for sampling images.
#     """
#     output_dir = os.path.join(output_dir, "sample")
#     os.makedirs(output_dir, exist_ok=True)

#     if sample_every_n_epochs is None:
#         sample_every_n_epochs = 0

#     if sample_every_n_steps is None:
#         sample_every_n_steps = 0

#     if sample_every_n_epochs == sample_every_n_steps == 0:
#         return run_cmd

#     # Create the prompt file and get its path
#     sample_prompts_path = os.path.join(output_dir, "prompt.txt")

#     with open(sample_prompts_path, "w") as f:
#         f.write(sample_prompts)

#     # Append the sampler with proper quoting for safety against special characters
#     run_cmd.append("--sample_sampler")
#     run_cmd.append(shlex.quote(sample_sampler))

#     # Normalize and fix the path for the sample prompts, handle cross-platform path differences
#     sample_prompts_path = os.path.abspath(os.path.normpath(sample_prompts_path))
#     if os.name == "nt":  # Normalize path for Windows
#         sample_prompts_path = sample_prompts_path.replace("\\", "/")

#     # Append the sample prompts path
#     run_cmd.append('--sample_prompts')
#     run_cmd.append(sample_prompts_path)

#     # Append the sampling frequency for epochs, only if non-zero
#     if sample_every_n_epochs != 0:
#         run_cmd.append("--sample_every_n_epochs")
#         run_cmd.append(str(sample_every_n_epochs))

#     # Append the sampling frequency for steps, only if non-zero
#     if sample_every_n_steps != 0:
#         run_cmd.append("--sample_every_n_steps")
#         run_cmd.append(str(sample_every_n_steps))

#     return run_cmd



class SampleImages:
    """
    A class for managing the Gradio interface for sampling images during training.
    """

    def __init__(
        self,
        config: KohyaSSGUIConfig = {},
    ):
        """
        Initializes the SampleImages class.
        """
        self.config = config
        
        self.initialize_accordion()

    def initialize_accordion(self):
        """
        Initializes the accordion for the Gradio interface.
        """
        with gr.Row():
            self.sample_every_n_steps = gr.Number(
                label="Sample every n steps",
                value=self.config.get("samples.sample_every_n_steps", 0),
                precision=0,
                interactive=True,
            )
            self.sample_every_n_epochs = gr.Number(
                label="Sample every n epochs",
                value=self.config.get("samples.sample_every_n_epochs", 0),
                precision=0,
                interactive=True,
            )
            self.sample_sampler = gr.Dropdown(
                label="Sample sampler",
                choices=[
                    "ddim",
                    "pndm",
                    "lms",
                    "euler",
                    "euler_a",
                    "heun",
                    "dpm_2",
                    "dpm_2_a",
                    "dpmsolver",
                    "dpmsolver++",
                    "dpmsingle",
                    "k_lms",
                    "k_euler",
                    "k_euler_a",
                    "k_dpm_2",
                    "k_dpm_2_a",
                ],
                value=self.config.get("samples.sample_sampler", "euler_a"),
                interactive=True,
            )
        with gr.Row():
            self.sample_prompts = gr.Textbox(
                lines=5,
                label="Sample prompts",
                interactive=True,
                placeholder="masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28",
                info="Enter one sample prompt per line to generate multiple samples per cycle. Optional specifiers include: --w (width), --h (height), --d (seed), --l (cfg scale), --s (sampler steps) and --n (negative prompt). To modify sample prompts during training, edit the prompt.txt file in the samples directory.",
                value=self.config.get("samples.sample_prompts", ""),
            )
