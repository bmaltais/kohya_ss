import os
import gradio as gr
import subprocess
import time
import webbrowser

try:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow  # Attempt to import tensorflow to check if it is installed

    visibility = True
except ImportError:
    visibility = False

from easygui import msgbox
from threading import Thread, Event
from .custom_logging import setup_logging
from .common_gui import setup_environment


class TensorboardManager:
    DEFAULT_TENSORBOARD_PORT = 6006

    def __init__(self, logging_dir, headless: bool = False, wait_time=5):
        self.logging_dir = logging_dir
        self.headless = headless
        self.wait_time = wait_time
        self.tensorboard_proc = None
        self.tensorboard_port = os.environ.get(
            "TENSORBOARD_PORT", self.DEFAULT_TENSORBOARD_PORT
        )
        self.log = setup_logging()
        self.thread = None
        self.stop_event = Event()

        self.gradio_interface()

    def get_button_states(self, started=False):
        return gr.Button(
            visible=visibility and (not started or self.headless)
        ), gr.Button(visible=visibility and (started or self.headless))

    def open_tensorboard_url(self):
        tensorboard_url = f"http://localhost:{self.tensorboard_port}"
        self.log.info(f"Opening TensorBoard URL in browser: {tensorboard_url}")
        webbrowser.open(tensorboard_url)

    def start_tensorboard(self, logging_dir=None):
        if self.tensorboard_proc is not None:
            self.log.info(
                "Tensorboard is already running. Terminating existing process before starting new one..."
            )
            self.stop_tensorboard()

        if not os.path.exists(logging_dir) or not os.listdir(logging_dir):
            self.log.error(
                "Error: logging folder does not exist or does not contain logs."
            )
            msgbox(msg="Error: logging folder does not exist or does not contain logs.")
            return self.get_button_states(started=False)

        run_cmd = [
            "tensorboard",
            "--logdir",
            logging_dir,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.tensorboard_port),
        ]

        self.log.info(run_cmd)

        self.log.info("Starting TensorBoard on port {}".format(self.tensorboard_port))
        try:
            env = setup_environment()
            self.tensorboard_proc = subprocess.Popen(run_cmd, env=env)
        except Exception as e:
            self.log.error("Failed to start Tensorboard:", e)
            return self.get_button_states(started=False)

        if not self.headless:
            self.stop_event.clear()
            
            time.sleep(self.wait_time)
            if not self.stop_event.is_set():
                self.thread = Thread(target=self.open_tensorboard_url)
                self.thread.start()

        return self.get_button_states(started=True)

    def stop_tensorboard(self):
        if self.tensorboard_proc is not None:
            self.log.info("Stopping tensorboard process...")
            try:
                self.tensorboard_proc.terminate()
                self.tensorboard_proc = None
                self.log.info("...process stopped")
            except Exception as e:
                self.log.error("Failed to stop Tensorboard:", e)

        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()  # Wait for the thread to finish
            self.thread = None
            self.log.info("Thread terminated successfully.")

        return self.get_button_states(started=False)

    def gradio_interface(self):

        with gr.Row():
            button_start_tensorboard = gr.Button(
                value="Start tensorboard",
                elem_id="myTensorButton",
                visible=visibility,
            )
            button_stop_tensorboard = gr.Button(
                value="Stop tensorboard",
                visible=visibility and self.headless,
                elem_id="myTensorButtonStop",
            )
            button_open_tensorboard = gr.Button(
                value="Open tensorboard",
                elem_id="myTensorButton",
                visible=not visibility,
                link=f"http://localhost:{self.tensorboard_port}",
            )
            button_start_tensorboard.click(
                self.start_tensorboard,
                inputs=[self.logging_dir],
                outputs=[button_start_tensorboard, button_stop_tensorboard],
                show_progress=False,
            )
            button_stop_tensorboard.click(
                self.stop_tensorboard,
                outputs=[button_start_tensorboard, button_stop_tensorboard],
                show_progress=False,
            )
