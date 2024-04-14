import subprocess
import psutil
import time
import gradio as gr
import shlex
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


class CommandExecutor:
    """
    A class to execute and manage commands.
    """

    def __init__(self):
        """
        Initialize the CommandExecutor.
        """
        self.process = None
        self.run_state = gr.Textbox(value="", visible=False)

    def execute_command(self, run_cmd: str, **kwargs):
        """
        Execute a command if no other command is currently running.

        Parameters:
        - run_cmd (str): The command to execute.
        - **kwargs: Additional keyword arguments to pass to subprocess.Popen.
        """
        if self.process and self.process.poll() is None:
            log.info("The command is already running. Please wait for it to finish.")
        else:
            # for i, item in enumerate(run_cmd):
            #     log.info(f"{i}: {item}")

            # Reconstruct the safe command string for display
            command_to_run = ' '.join(run_cmd)
            log.info(f"Executings command: {command_to_run}")

            # Execute the command securely
            self.process = subprocess.Popen(run_cmd, **kwargs)

    def kill_command(self):
        """
        Kill the currently running command and its child processes.
        """
        if self.is_running():
            try:
                # Get the parent process and kill all its children
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                log.info("The running process has been terminated.")
            except psutil.NoSuchProcess:
                # Explicitly handle the case where the process does not exist
                log.info(
                    "The process does not exist. It might have terminated before the kill command was issued."
                )
            except Exception as e:
                # General exception handling for any other errors
                log.info(f"Error when terminating process: {e}")
        else:
            log.info("There is no running process to kill.")
            
        return gr.Button(visible=True), gr.Button(visible=False)
    
    def wait_for_training_to_end(self):
        while self.is_running():
            time.sleep(1)
            log.debug("Waiting for training to end...")
        log.info("Training has ended.")
        return gr.Button(visible=True), gr.Button(visible=False)

    def is_running(self):
        """
        Check if the command is currently running.

        Returns:
        - bool: True if the command is running, False otherwise.
        """
        return self.process and self.process.poll() is None