class _DummySafe:
    def __init__(self):
        self._keys = []

    def keys(self):
        return self._keys


def safe_open(filename=None, framework=None, device=None):
    # Return a dummy object with keys() for tests
    return _DummySafe()
