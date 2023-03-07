import tempfile
import gradio as gr
from easygui import msgbox

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄


###
### Gradio common sampler GUI section
###


def sample_gradio_config():
    with gr.Accordion('Sample images config', open=False):
        with gr.Row():
            sample_every_n_steps = gr.Number(
                label='Sample every n steps',
                value=0,
                precision=0,
                interactive=True,
            )
            sample_every_n_epochs = gr.Number(
                label='Sample every n epochs',
                value=0,
                precision=0,
                interactive=True,
            )
            sample_sampler = gr.Dropdown(
                label='Sample sampler',
                choices=[
                    'ddim',
                    'pndm',
                    'lms',
                    'euler',
                    'euler_a',
                    'heun',
                    'dpm_2',
                    'dpm_2_a',
                    'dpmsolver',
                    'dpmsolver++',
                    'dpmsingle',
                    'k_lms',
                    'k_euler',
                    'k_euler_a',
                    'k_dpm_2',
                    'k_dpm_2_a',
                ],
                value='euler_a',
                interactive=True,
            )
        with gr.Row():
            sample_prompts = gr.Textbox(
                lines=5,
                label='Sample prompts',
                interactive=True,
            )
    return (
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
    )


def run_cmd_sample(
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
):
    run_cmd = ''
    
    if sample_every_n_epochs == 0 and sample_every_n_steps == 0:
        return run_cmd

    # Create a temporary file and get its path
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        # Write the contents of the variable to the file
        temp_file.write(sample_prompts)

    # Get the path of the temporary file
    sample_prompts_path = temp_file.name

    run_cmd += f' --sample_sampler={sample_sampler}'
    run_cmd += f' --sample_prompts="{sample_prompts_path}"'

    if not sample_every_n_epochs == 0:
        run_cmd += f' --sample_every_n_epochs="{sample_every_n_epochs}"'

    if not sample_every_n_steps == 0:
        run_cmd += f' --sample_every_n_steps="{sample_every_n_steps}"'

    return run_cmd
