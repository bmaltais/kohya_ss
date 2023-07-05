import tempfile
import os
import gradio as gr
from easygui import msgbox

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

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
                info='这个参数决定了在训练过程中，每隔多少步骤进行一次采样。采样在这里意味着训练模型生成一些输出，以便您可以观察模型的性能。例如，如果将此值设置为100，那么模型将每训练100步生成一次样本。如果将此值设置为0，那么模型将不会在训练过程中进行任何采样'
            )
            sample_every_n_epochs = gr.Number(
                label='Sample every n epochs',
                value=0,
                precision=0,
                interactive=True,
                info='这个参数与Sample every n steps类似，只是它以训练的"epoch"数为单位进行采样。"Epoch"是指模型处理完整个数据集一次的过程'
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
                placeholder='masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28',
                info='提供一个或多个文本提示，模型将在训练过程中根据这些提示生成样本。这个参数是非常灵活的，你可以使用它来引导模型生成特定类型的输出，或者来测试模型对特定提示的响应。例如，你在这个参数中输入的示例提示包含了一些描述艺术品质量的词，这可能表明你希望模型生成一些与这些词相关的艺术品样本'
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
    output_dir,
):
    output_dir = os.path.join(output_dir, 'sample')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_cmd = ''

    if sample_every_n_epochs == 0 and sample_every_n_steps == 0:
        return run_cmd

    # Create the prompt file and get its path
    sample_prompts_path = os.path.join(output_dir, 'prompt.txt')

    with open(sample_prompts_path, 'w') as f:
        f.write(sample_prompts)

    run_cmd += f' --sample_sampler={sample_sampler}'
    run_cmd += f' --sample_prompts="{sample_prompts_path}"'

    if not sample_every_n_epochs == 0:
        run_cmd += f' --sample_every_n_epochs="{sample_every_n_epochs}"'

    if not sample_every_n_steps == 0:
        run_cmd += f' --sample_every_n_steps="{sample_every_n_steps}"'

    return run_cmd
