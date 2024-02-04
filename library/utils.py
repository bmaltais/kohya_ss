import logging
import sys
import threading
from typing import *


def fire_in_thread(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()


def add_logging_arguments(parser):
    parser.add_argument(
        "--console_log_level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level, default is INFO / ログレベルを設定する。デフォルトはINFO",
    )
    parser.add_argument(
        "--console_log_file",
        type=str,
        default=None,
        help="Log to a file instead of stdout / 標準出力ではなくファイルにログを出力する",
    )
    parser.add_argument("--console_log_simple", action="store_true", help="Simple log output / シンプルなログ出力")


def setup_logging(args=None, log_level=None, reset=False):
    if logging.root.handlers:
        if reset:
            # remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    if log_level is None and args is not None:
        log_level = args.console_log_level
    if log_level is None:
        log_level = "INFO"
    log_level = getattr(logging, log_level)

    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        handler = None
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler

                handler = RichHandler()
            except ImportError:
                print("rich is not installed, using basic logging")

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)  # same as print
            handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)
