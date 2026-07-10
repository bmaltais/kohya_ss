import collections
import subprocess
import sys
import threading
import time
from typing import Deque, Optional, TextIO, Tuple

import gradio as gr
import psutil

from .common_gui import output_message
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# How many trailing log lines to attach to a failure summary.
_DEFAULT_LOG_TAIL_LINES = 40


def tail_lines(text: Optional[str], max_lines: int = _DEFAULT_LOG_TAIL_LINES) -> str:
    """Return the last ``max_lines`` lines of ``text`` (empty-safe)."""
    if not text:
        return ""
    lines = text.splitlines()
    if max_lines <= 0:
        return ""
    return "\n".join(lines[-max_lines:])


def read_log_tail(path: str, max_lines: int = _DEFAULT_LOG_TAIL_LINES) -> str:
    """Read the last ``max_lines`` lines from a log file; empty if missing."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return tail_lines(f.read(), max_lines=max_lines)
    except OSError:
        return ""


def drain_stream_to_buffer(
    stream: TextIO,
    buffer: Deque[str],
    *,
    dest: Optional[TextIO] = None,
) -> None:
    """Read lines from ``stream`` into ``buffer``, optionally echoing to ``dest``.

    Used to tee subprocess output so a short failure tail is available without
    hiding live console logs. Safe to call on a background thread.
    """
    try:
        for line in iter(stream.readline, ""):
            if dest is not None:
                try:
                    dest.write(line)
                    dest.flush()
                except OSError:
                    pass
            buffer.append(line.rstrip("\n\r"))
    except (OSError, ValueError):
        # Stream closed underfoot (process exit) — ignore.
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def summarize_training_end(
    returncode: Optional[int],
    *,
    user_stopped: bool = False,
    log_tail: str = "",
) -> Tuple[str, str]:
    """Build a (level, message) pair for end-of-training status.

    Levels: ``info`` or ``error``. User-initiated stop is always reported
    as cancelled (info), even when the OS exit code is non-zero.
    """
    if user_stopped:
        return "info", "Training was stopped by the user."

    if returncode is None:
        return "info", "Training has ended."

    if returncode == 0:
        return "info", "Training has ended successfully (exit code 0)."

    msg = f"Training failed with exit code {returncode}."
    if log_tail and log_tail.strip():
        msg = f"{msg}\n\nLast log lines:\n{log_tail.strip()}"
    return "error", msg


class CommandExecutor:
    """
    A class to execute and manage commands.
    """

    def __init__(self, headless: bool = False):
        """
        Initialize the CommandExecutor.
        """
        self.headless = headless
        self.process = None
        self._stopped_by_user = False
        self._output_lines: Deque[str] = collections.deque(
            maxlen=_DEFAULT_LOG_TAIL_LINES
        )
        self._tee_thread: Optional[threading.Thread] = None

        with gr.Row():
            self.button_run = gr.Button("Start training", variant="primary")

            self.button_stop_training = gr.Button(
                "Stop training",
                visible=self.process is not None or headless,
                variant="stop",
            )

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
            # Reconstruct the safe command string for display
            command_to_run = " ".join(run_cmd)
            log.info(f"Executing command: {command_to_run}")

            self._stopped_by_user = False
            self._output_lines = collections.deque(maxlen=_DEFAULT_LOG_TAIL_LINES)
            self._tee_thread = None

            # Tee child stdout/stderr into a ring buffer (and the console) so
            # failure summaries can include a traceback tail. Skip if the
            # caller already redirected either stream.
            tee_output = "stdout" not in kwargs and "stderr" not in kwargs
            popen_kwargs = dict(kwargs)
            if tee_output:
                popen_kwargs["stdout"] = subprocess.PIPE
                popen_kwargs["stderr"] = subprocess.STDOUT
                popen_kwargs.setdefault("text", True)
                popen_kwargs.setdefault("encoding", "utf-8")
                popen_kwargs.setdefault("errors", "replace")
                # Line-buffered text mode when supported.
                popen_kwargs.setdefault("bufsize", 1)

            # Execute the command securely
            self.process = subprocess.Popen(run_cmd, **popen_kwargs)
            log.debug("Command executed.")

            if tee_output and self.process.stdout is not None:
                self._tee_thread = threading.Thread(
                    target=drain_stream_to_buffer,
                    args=(self.process.stdout, self._output_lines),
                    kwargs={"dest": sys.stdout},
                    name="command-executor-tee",
                    daemon=True,
                )
                self._tee_thread.start()

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
                self._stopped_by_user = True
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
            self.process = None
            log.info("There is no running process to kill.")

        return gr.Button(visible=True), gr.Button(visible=False or self.headless)

    def _join_tee(self, timeout: float = 2.0) -> None:
        thread = self._tee_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _collect_log_tail(self) -> str:
        """Prefer teed process lines; fall back to GUI setup.log."""
        self._join_tee()
        if self._output_lines:
            return "\n".join(self._output_lines)
        # custom_logging writes the GUI log here (cwd-relative).
        return read_log_tail("setup.log", max_lines=_DEFAULT_LOG_TAIL_LINES)

    def wait_for_training_to_end(self):
        while self.is_running():
            time.sleep(1)
            log.debug("Waiting for training to end...")

        returncode: Optional[int] = None
        if self.process is not None:
            # Ensure returncode is populated after the process has exited.
            returncode = self.process.poll()
            if returncode is None:
                returncode = self.process.returncode

        include_tail = returncode not in (None, 0) and not self._stopped_by_user
        log_tail = self._collect_log_tail() if include_tail else ""
        level, message = summarize_training_end(
            returncode,
            user_stopped=self._stopped_by_user,
            log_tail=log_tail,
        )

        if level == "error":
            log.error(message)
            output_message(msg=message, title="Training failed", headless=self.headless)
        else:
            log.info(message)

        return gr.Button(visible=True), gr.Button(visible=False or self.headless)

    def is_running(self):
        """
        Check if the command is currently running.

        Returns:
        - bool: True if the command is running, False otherwise.
        """
        return self.process is not None and self.process.poll() is None
