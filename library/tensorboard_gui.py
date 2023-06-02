import os
import gradio as gr
from easygui import msgbox
import subprocess
import time

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

tensorboard_proc = None   # I know... bad but heh
TENSORBOARD = 'tensorboard' if os.name == 'posix' else 'tensorboard.exe'


def start_tensorboard(logging_dir):
    global tensorboard_proc

    if not os.listdir(logging_dir):
        log.info('Error: log folder is empty')
        msgbox(msg='Error: log folder is empty')
        return

    run_cmd = [f'{TENSORBOARD}', '--logdir', f'{logging_dir}']

    log.info(run_cmd)
    if tensorboard_proc is not None:
        log.info(
            'Tensorboard is already running. Terminating existing process before starting new one...'
        )
        stop_tensorboard()

    # Start background process
    log.info('Starting tensorboard...')
    tensorboard_proc = subprocess.Popen(run_cmd)

    # Wait for some time to allow TensorBoard to start up
    time.sleep(5)

    # Open the TensorBoard URL in the default browser
    log.info('Opening tensorboard url in browser...')
    import webbrowser

    webbrowser.open('http://localhost:6006')


def stop_tensorboard():
    log.info('Stopping tensorboard process...')
    tensorboard_proc.kill()
    log.info('...process stopped')


def gradio_tensorboard():
    with gr.Row():
        button_start_tensorboard = gr.Button('Start tensorboard')
        button_stop_tensorboard = gr.Button('Stop tensorboard')

    return (button_start_tensorboard, button_stop_tensorboard)
