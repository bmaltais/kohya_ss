import gradio as gr
import os
import argparse
from dreambooth_gui import dreambooth_tab
from finetune_gui import finetune_tab
from library.utilities import utilities_tab
from library.extract_lora_gui import gradio_extract_lora_tab
from library.merge_lora_gui import gradio_merge_lora_tab
from lora_gui import lora_tab


def UI(username, password):

    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css)

    with interface:
        with gr.Tab('Dreambooth'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab()
        with gr.Tab('Dreambooth LoRA'):
            lora_tab()
        with gr.Tab('Finetune'):
            finetune_tab()
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
            )
            gradio_extract_lora_tab()
            gradio_merge_lora_tab()

    # Show the interface
    if not username == '':
        interface.launch(auth=(username, password))
    else:
        interface.launch()


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )

    args = parser.parse_args()

    UI(username=args.username, password=args.password)
