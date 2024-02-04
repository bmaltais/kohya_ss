import logging
import sys
import threading
from typing import *


def fire_in_thread(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()


def setup_logging(log_level=logging.INFO):
    if logging.root.handlers:  # Already configured
        return

    try:
        from rich.logging import RichHandler

        handler = RichHandler()
    except ImportError:
        print("rich is not installed, using basic logging")
        handler = logging.StreamHandler(sys.stdout)  # same as print
        handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)
