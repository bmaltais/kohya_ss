import os
import gradio as gr
from easygui import msgbox
import subprocess
import time

tensorboard_proc = None # I know... bad but heh

def start_tensorboard(logging_dir):
    global tensorboard_proc
    
    if not os.listdir(logging_dir):
        print("Error: log folder is empty")
        msgbox(msg="Error: log folder is empty")
        return
    
    run_cmd = f'tensorboard.exe --logdir "{logging_dir}"'
    
    print(run_cmd)
    if tensorboard_proc is not None:
        print("Tensorboard is already running. Terminating existing process before starting new one...")
        stop_tensorboard()
    
    # Start background process
    print('Starting tensorboard...')    
    tensorboard_proc = subprocess.Popen(run_cmd)
    
    # Wait for some time to allow TensorBoard to start up
    time.sleep(5)
    
    # Open the TensorBoard URL in the default browser
    print('Opening tensorboard url in browser...')
    import webbrowser
    webbrowser.open('http://localhost:6006')
    
def stop_tensorboard():
    print('Stopping tensorboard process...')
    tensorboard_proc.kill()
    print('...process stopped')
    
def gradio_tensorboard():
    with gr.Row():
        button_start_tensorboard = gr.Button('Start tensorboard')
        button_stop_tensorboard = gr.Button('Stop tensorboard')
        
    return(button_start_tensorboard, button_stop_tensorboard)
