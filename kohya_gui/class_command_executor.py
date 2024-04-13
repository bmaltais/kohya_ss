import subprocess
import psutil
import os
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
            if os.name == 'nt':
                run_cmd = run_cmd.replace('\\', '/')
                
            # Split the command string into components
            parts = shlex.split(run_cmd)
            # The first part is the executable, and it doesn't need quoting
            executable = parts[0]
            
            # The remaining parts are the arguments, which we will quote for safety
            safe_args = [shlex.quote(part) for part in parts[1:]]
            
            # Log the executable and arguments to debug path issues
            log.info(f"Executable: {executable}")
            log.info(f"Arguments: {' '.join(safe_args)}")
            
            # Reconstruct the safe command string for display
            command_to_run = ' '.join([executable] + safe_args)
            log.info(f"Executing command: {command_to_run}")

            # Execute the command securely
            self.process = subprocess.Popen([executable] + safe_args, **kwargs)

    def kill_command(self):
        """
        Kill the currently running command and its child processes.
        """
        if self.process and self.process.poll() is None:
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
