class Blocks:
    def __init__(self, *args, **kwargs):
        # simple stub: remember launch kwargs
        self._launched_kwargs = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def launch(self, **kwargs):
        self._launched_kwargs = kwargs


class Tab(Blocks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __enter__(self):
        return self


def Markdown(*args, **kwargs):
    # no-op for tests
    return None


# Generic fallback for common Gradio component classes used only for type
# annotations or simple construction in tests. Return a no-op class for any
# attribute access like gr.Checkbox, gr.Number, etc.
class _Component:
    def __init__(self, *args, **kwargs):
        pass


def __getattr__(name):
    return _Component
