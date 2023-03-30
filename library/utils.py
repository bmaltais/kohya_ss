import threading
from typing import *


def fire_in_thread(f):
    def wrapped(*args, **kwargs):
        threading.Thread(target=f, args=args, kwargs=kwargs).start()
    return wrapped