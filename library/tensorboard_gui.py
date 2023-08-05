import os
import gradio as gr
from easygui import msgbox
import subprocess
import time
import webbrowser
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

tensorboard_proc = None
TENSORBOARD = 'tensorboard' if os.name == 'posix' else 'tensorboard.exe'

def start_tensorboard(logging_dir, wait_time=5):
    global tensorboard_proc

    if not os.listdir(logging_dir):
        log.info('Error: log folder is empty')
        msgbox(msg='Error: log folder is empty')
        return

    run_cmd = [TENSORBOARD, '--logdir', logging_dir, '--host', '0.0.0.0', '--port', '6006']

    log.info(run_cmd)
    if tensorboard_proc is not None:
        log.info('Tensorboard is already running. Terminating existing process before starting new one...')
        stop_tensorboard()

    # Start background process
    log.info('Starting tensorboard...')
    try:
        tensorboard_proc = subprocess.Popen(run_cmd)
    except Exception as e:
        log.error('Failed to start Tensorboard:', e)
        return

    # Wait for some time to allow TensorBoard to start up
    time.sleep(wait_time)

    # Open the TensorBoard URL in the default browser
    log.info('Opening tensorboard url in browser...')
    webbrowser.open('http://localhost:6006')

def stop_tensorboard():
    global tensorboard_proc
    if tensorboard_proc is not None:
        log.info('Stopping tensorboard process...')
        try:
            tensorboard_proc.terminate()
            tensorboard_proc = None
            log.info('...process stopped')
        except Exception as e:
            log.error('Failed to stop Tensorboard:', e)
    else:
        log.info('Tensorboard is not running...')

def gradio_tensorboard():
    with gr.Row():
        button_start_tensorboard = gr.Button('Start tensorboard')
        button_stop_tensorboard = gr.Button('Stop tensorboard')

    return (button_start_tensorboard, button_stop_tensorboard)
