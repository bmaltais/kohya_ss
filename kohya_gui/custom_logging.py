import os
import logging
from logging.handlers import RotatingFileHandler
import time
import sys
import toml

from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install

log = None

DEFAULT_LOGGING_CONFIG = {
    "console_log_level": "INFO",
    "file_log_level": "DEBUG",
    "log_file_name": "setup.log",
    "max_log_file_size": 10 * 1024 * 1024,  # 10 MB
    "log_backup_count": 5,
}

def get_logging_level(level_str: str, default_level: int) -> int:
    """Converts a log level string to its logging module integer equivalent."""
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return level_map.get(level_str.upper(), default_level)

def setup_logging(clean=False):
    global log

    if log is not None:
        return log

    # Load configuration
    config = DEFAULT_LOGGING_CONFIG.copy()
    try:
        # Adjust path to be relative to this file's location
        config_file_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
        loaded_config = toml.load(config_file_path)
        if "logging" in loaded_config:
            for key, value in DEFAULT_LOGGING_CONFIG.items():
                config[key] = loaded_config["logging"].get(key, value)
    except FileNotFoundError:
        # Keep default config if config.toml doesn't exist
        pass
    except Exception as e:
        # Log potential errors during config loading but proceed with defaults
        # This uses a temporary basic logger setup in case the main one fails.
        temp_logger = logging.getLogger("custom_logging_setup_error")
        temp_logger.addHandler(logging.StreamHandler(sys.stderr)) # Log to stderr
        temp_logger.error(f"Error loading logging config from TOML: {e}. Using default logging settings.")


    log_file_name = config.get("log_file_name", DEFAULT_LOGGING_CONFIG["log_file_name"])
    file_log_level_str = config.get("file_log_level", DEFAULT_LOGGING_CONFIG["file_log_level"])
    console_log_level_str = config.get("console_log_level", DEFAULT_LOGGING_CONFIG["console_log_level"])

    file_log_level = get_logging_level(file_log_level_str, logging.DEBUG)
    console_log_level = get_logging_level(console_log_level_str, logging.INFO)
    max_log_file_size = config.get("max_log_file_size", DEFAULT_LOGGING_CONFIG["max_log_file_size"])
    log_backup_count = config.get("log_backup_count", DEFAULT_LOGGING_CONFIG["log_backup_count"])

    log = logging.getLogger("sd") # Get the logger instance
    log.setLevel(min(file_log_level, console_log_level)) # Set logger to the more verbose level

    try:
        if clean and os.path.isfile(log_file_name):
            os.remove(log_file_name)
        time.sleep(0.1)  # prevent race condition
    except:
        pass

    # Configure RotatingFileHandler
    file_handler = RotatingFileHandler(
        filename=log_file_name,
        maxBytes=max_log_file_size,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(pathname)s | %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(file_log_level)
    log.addHandler(file_handler)

    # Configure RichHandler for console
    console = Console(
        log_time=True,
        log_time_format="%H:%M:%S-%f",
        theme=Theme(
            {
                "traceback.border": "black",
                "traceback.border.syntax_error": "black",
                "inspect.value.border": "black",
            }
        ),
    )
    pretty_install(console=console)
    traceback_install(
        console=console,
        extra_lines=1,
        width=console.width,
        word_wrap=False,
        indent_guides=False,
        suppress=[],
    )
    rh = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format="%H:%M:%S-%f",
        level=console_log_level,
        console=console,
    )
    # log = logging.getLogger("sd") # Already initialized above
    log.addHandler(rh)

    return log
