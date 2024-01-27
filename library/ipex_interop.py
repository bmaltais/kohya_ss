import torch


def init_ipex():
    """
    Try to import `intel_extension_for_pytorch`, and apply
    the hijacks using `library.ipex.ipex_init`.

    If IPEX is not installed, this function does nothing.
    """
    try:
        import intel_extension_for_pytorch as ipex  # noqa
    except ImportError:
        return

    try:
        from library.ipex import ipex_init

        if torch.xpu.is_available():
            is_initialized, error_message = ipex_init()
            if not is_initialized:
                print("failed to initialize ipex:", error_message)
    except Exception as e:
        print("failed to initialize ipex:", e)
